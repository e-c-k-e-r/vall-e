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

if not distributed_initialized() and cfg.trainer.backend == "deepspeed":
	init_distributed(init_deepspeed_dist)

class Engine(DeepSpeedEngine):
	def __init__(self, *args, **kwargs):
		if '_cfg' in kwargs:
			self._cfg = kwargs['_cfg']
			kwargs.pop("_cfg")

		kwargs['config'] = cfg.trainer.deepspeed.ds_cfg
		kwargs['config_class'] = DeepSpeedConfig(kwargs['config'])

		super().__init__(None, *args, **kwargs)
		self._frozen_params = set()

		self.tokens_processed = 0

	def freeze(self):
		for p in self.module.parameters():
			if p.requires_grad:
				p.requires_grad_(False)
				self._frozen_params.add(p)

	def unfreeze(self):
		for p in self._frozen_params:
			p.requires_grad_(True)
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
					param_group['lr'] = lr
			else:
				self.optimizer.set_lr(lr)
		except Exception as e:
			print(str(e))

	def traverse(self, *args, **kwargs):
		with torch.autocast(self.device, dtype=cfg.trainer.dtype, enabled=cfg.trainer.amp):
			self.forward(*args, **kwargs)
			losses = self.gather_attribute("loss")
			loss = torch.stack([*losses.values()]).sum()

		stats = {}
		stats |= {k: v.item() for k, v in losses.items()}
		stats |= self.gather_attribute("scalar")

		self.backward(loss)
		self.step()

		return stats