"""
# https://github.com/enhuiz/pytorch-training-utilities
"""

# to-do: replace this
# to-do: swap out deepspeed

from ..config import cfg
from ..utils import dispatch_attribute, flatten_dict, gather_attribute, do_gc, to_device

import logging
import time
import torch
import torch.distributed

from torch import Tensor
from torch.distributed import all_reduce
from typing import Any, Protocol

from .base import TrainFeeder

_logger = logging.getLogger(__name__)

from deepspeed import DeepSpeedEngine, DeepSpeedConfig, comm as dist, init_distributed as init_deepspeed_dist
from deepspeed.accelerator import get_accelerator

from ..utils.distributed import init_distributed, distributed_initialized
from ..utils import wrapper as ml

if not distributed_initialized() and cfg.trainer.backend == "deepspeed":
	init_distributed(init_deepspeed_dist)

class Engine(DeepSpeedEngine):
	def __init__(self, *args, **kwargs):
		self._cfg = None
		if '_cfg' in kwargs:
			self._cfg = kwargs['_cfg']
			kwargs.pop("_cfg")

		kwargs['config'] = cfg.trainer.deepspeed.ds_cfg
		kwargs['config_class'] = DeepSpeedConfig(kwargs['config'])

		stats = {
			"global_step": 0,
			"micro_step": 0,
			"global_samples": 0,
			"tokens_processed": 0,
		}

		# kwargs['stats'] = None will return None when popped
		maybe_stats = kwargs.pop('stats', stats)
		if maybe_stats is not None:
			stats = maybe_stats

		super().__init__(None, *args, **kwargs)
		self._frozen_params = set()

		self.global_steps = stats["global_step"]
		self.micro_steps = stats["micro_step"]
		self.global_samples = stats["global_samples"]
		self.tokens_processed = stats["tokens_processed"]

		self.max_nan_losses = 8

	def freeze(self, freeze_all=True):
		if self._cfg is None or not hasattr(self._cfg, "frozen_params"):
			raise Exception("freeze_all=False yet self._cfg.frozen_params is None")

		for name, param in self.module.named_parameters():
			if (freeze_all and param.requires_grad) or (not freeze_all and name in self._cfg.frozen_params):
				param.requires_grad_(False)
				self._frozen_params.add(param)

	def unfreeze(self):
		for param in self._frozen_params:
			param.requires_grad_(True)
		self._frozen_params.clear()
	
	@property
	def _training(self):
		return self._cfg.training

	@property
	def global_step(self):
		return self.global_steps

	@property
	def micro_step(self):
		return self.micro_steps	

	@property
	def batch_size(self):
		return cfg.hyperparameters.batch_size

	def gather_attribute(self, *args, **kwargs):
		return gather_attribute(self.module, *args, **kwargs)

	def dispatch_attribute(self, *args, **kwargs):
		return dispatch_attribute(self.module, *args, **kwargs)

	def set_lr(self, lr):
		try:
			if hasattr(self.optimizer, 'param_groups'):
				for param_group in self.optimizer.param_groups:
					param_group["d_coeff" if "d_coeff" in param_group else "lr"] = lr
			else:
				self.optimizer.set_lr(lr)
		except Exception as e:
			print(str(e))

	def traverse(self, *args, **kwargs):
		with ml.autocast():
			self.forward(*args, **kwargs)

		losses = self.gather_attribute("loss")
		loss = torch.stack([*losses.values()]).sum()

		if torch.isnan(loss).any():
			self.max_nan_losses = self.max_nan_losses - 1
			if self.max_nan_losses < 0:
				raise RuntimeError("Too many NaN losses detected.")

		stats = {}
		stats |= {k: v.item() for k, v in losses.items()}
		stats |= self.gather_attribute("scalar")

		self.backward(loss)
		self.step()

		return stats