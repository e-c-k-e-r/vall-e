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
from ..utils import ml

from ..models.lora import freeze_non_lora_weights

if not distributed_initialized() and cfg.trainer.backend == "deepspeed":
	init_distributed(init_deepspeed_dist)

class Engine(DeepSpeedEngine):
	def __init__(self, *args, **kwargs):
		self.hyper_config = kwargs.pop('hyper_config', None)

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

		self.global_steps = stats["global_step"]
		self.micro_steps = stats["micro_step"]
		self.global_samples = stats["global_samples"]
		self.tokens_processed = stats["tokens_processed"]

		self._frozen_params = set()
		self.current_batch_size = 0

	def freeze(self, freeze_all=True):
		# freeze non-LoRA params if requested
		if not self.hyper_config.frozen_params and not freeze_all and cfg.lora is not None:
			frozen_params = freeze_non_lora_weights( self.module, embeddings=cfg.lora.embeddings )
			for param in frozen_params:
				self._frozen_params.add( param )

			return

		if self.hyper_config is None or not hasattr(self.hyper_config, "frozen_params"):
			raise Exception("freeze_all=False yet self.hyper_config.frozen_params is None")

		for name, param in self.module.named_parameters():
			if (freeze_all and param.requires_grad) or (not freeze_all and name in self.hyper_config.frozen_params):
				param.requires_grad_(False)
				self._frozen_params.add(param)

	def unfreeze(self):
		for param in self._frozen_params:
			param.requires_grad_(True)
		self._frozen_params.clear()
	
	@property
	def _training(self):
		return self.hyper_config.training

	@property
	def _teacher(self):
		return self.hyper_config.teacher

	@property
	def global_step(self):
		return self.global_steps

	@property
	def micro_step(self):
		return self.micro_steps	

	@property
	def batch_size(self):
		return self.current_batch_size if self.current_batch_size > 0 else cfg.hyperparameters.batch_size

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
			_logger.warning(str(e))

	# cur_scale, because _get_loss_scale has a typo in the def and I can't be assed to inject a fix into it or push a PR
	def get_loss_scale(self):
		if not hasattr(self.optimizer, "cur_scale") or self.optimizer.cur_scale is None:
			return 1.0

		return self.optimizer.cur_scale

	def set_loss_scale(self, value):
		if not hasattr(self.optimizer, "cur_scale") or self.optimizer.cur_scale is None:
			return
		
		self.optimizer.cur_scale = value

	# we'll just have to live with the LoRA weights living within our main weights
	# they're easy to extract anyways
	def load_checkpoint(self, load_dir, **kwargs ):
		# override to load the lora instead
		if cfg.lora is not None:
			load_dir = cfg.ckpt_dir / cfg.lora.full_name

		return super().load_checkpoint( load_dir, **kwargs )

	def save_checkpoint(self, save_dir, **kwargs ):
		# override to save the lora instead
		if cfg.lora is not None:
			save_dir = cfg.ckpt_dir / cfg.lora.full_name

		return super().save_checkpoint( save_dir, **kwargs )

	def traverse(self, *args, **kwargs):
		with ml.autocast():
			self.forward(*args, **kwargs)

		losses = self.gather_attribute("loss")
		loss = torch.stack([*losses.values()]).sum()

		stats = {}
		stats |= {k: v.item() for k, v in losses.items()}
		stats |= self.gather_attribute("scalar")

		"""
		if torch.isnan(loss).any():
			self.max_nan_losses = self.max_nan_losses - 1
			if self.max_nan_losses < 0:
				raise RuntimeError("Too many NaN losses detected.")
			
			return stats
		"""

		self.backward(loss)
		self.step()

		return stats