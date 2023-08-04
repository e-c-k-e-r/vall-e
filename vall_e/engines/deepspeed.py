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

from deepspeed import DeepSpeedEngine, DeepSpeedConfig, comm as dist, init_distributed
from deepspeed.accelerator import get_accelerator

#dist.init_distributed(dist_backend=get_accelerator().communication_backend_name())
initialized_dist = False
if not initialized_dist:
	initialized_dist = True
	init_distributed()

class Engine(DeepSpeedEngine):
	def __init__(self, *args, **kwargs):
		kwargs['config'] = cfg.trainer.deepspeed.get_ds_cfg(model=kwargs['model'])
		kwargs['config_class'] = DeepSpeedConfig(kwargs['config'])

		super().__init__(None, *args, **kwargs)
		self._frozen_params = set()

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
	def global_step(self):
		return self.global_steps

	@property
	def micro_step(self):
		return self.micro_steps

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
		self.forward(*args, **kwargs)
		losses = self.gather_attribute("loss")
		loss = torch.stack([*losses.values()]).sum()

		stats = {}
		stats |= {k: v.item() for k, v in losses.items()}
		stats |= self.gather_attribute("scalar")

		self.backward(loss)
		self.step()

		return stats