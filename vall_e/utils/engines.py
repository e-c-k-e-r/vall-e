"""
# https://github.com/enhuiz/pytorch-training-utilities
"""

# to-do: replace this
# to-do: swap out deepspeed

from ..config import Config
from .distributed import fix_unset_envs
from .utils import dispatch_attribute, flatten_dict, gather_attribute, do_gc, to_device

import logging
import time
import torch
import torch.distributed

from deepspeed import DeepSpeedEngine
from torch import Tensor
from torch.distributed import all_reduce
from typing import Any, Protocol

Stats = dict[str, float]

_logger = logging.getLogger(__name__)


class Engine(DeepSpeedEngine):
	def __init__(self, *args, **kwargs):
		fix_unset_envs()
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

	def gather_attribute(self, *args, **kwargs):
		return gather_attribute(self.module, *args, **kwargs)

	def dispatch_attribute(self, *args, **kwargs):
		return dispatch_attribute(self.module, *args, **kwargs)


class TrainFeeder(Protocol):
	def __call__(
		self, *, engines: "Engines", batch: Any, name: str
	) -> None | tuple[Tensor, Stats]:
		...


class Engines(dict[str, Engine]):
	def setup(self, cfg: Config):
		self._cfg = cfg
		self._global_step = 0

	@property
	def cfg(self) -> Config:
		return self._cfg

	@property
	def config(self):
		return self._cfg

	@property
	def global_step(self):
		return self._global_step

	def gather_attribute(self, *args, **kwargs):
		ret = {}
		for engine in self.values():
			ret |= engine.gather_attribute(*args, **kwargs)
		return ret

	def dispatch_attribute(self, *args, **kwargs):
		for engine in self.values():
			engine.dispatch_attribute(*args, **kwargs)

	def save_checkpoint(self, tag=None):
		if not tag:
			tag = self.cfg.trainer.save_tag
		tag = tag.lower()
		if tag[:2] == "it" or tag[:4] == "step":
			tag = self.global_step

		self.cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
		for name, engine in self.items():
			engine.save_checkpoint(self.cfg.ckpt_dir / name, tag=tag)

	def load_checkpoint(self, tag=None):
		if not tag:
			tag = self.cfg.trainer.load_tag

		for name, engine in self.items():
			load_dir = self.cfg.ckpt_dir / name
			engine.load_checkpoint(
				tag=tag,
				load_dir=load_dir,
				load_module_strict=self.cfg.trainer.strict_loading,
				load_optimizer_states=self.cfg.trainer.load_states,
				load_lr_scheduler_states=self.cfg.trainer.load_states,
				load_module_only=False, # not self.cfg.trainer.load_states,
			)
			if self.cfg.trainer.restart_step_count:
				engine.global_steps = 0

		# update the LR because for some god awful reason it gets overwritten when loading from a checkpoint but only when it's not using a scheduler
		if self.cfg.hyperparameters.scheduler_type == "":
			self.set_lr(self.cfg.hyperparameters.learning_rate)

		self._update_global_step()

	def set_lr(self, lr):
		try:
			for engine in self.values():
				if hasattr(engine.optimizer, 'param_groups'):
					print(engine.optimizer.param_groups)
					for param_group in engine.optimizer.param_groups:
						param_group['lr'] = lr
				else:
					engine.optimizer.set_lr(lr)
		except Exception as e:
			print(str(e))

	def _update_global_step(self):
		for engine in self.values():
			self._global_step = max(self._global_step, engine.global_step)

	def eval(self):
		for engine in self.values():
			engine.eval()

	def train(self):
		for engine in self.values():
			engine.train()

	def step(self, feeder: TrainFeeder, batch):
		total_elapsed_time = 0

		stats: Any = dict()

		if self.cfg.trainer.gc_mode == 'step':
			do_gc()

		batch = to_device(batch, torch.cuda.current_device())

		for name, engine in self.items():
			torch.cuda.synchronize()
			if self.cfg.trainer.gc_mode == 'substep':
				do_gc()

			start_time = time.time()

			tries = 4
			n_ooms = torch.zeros([], device=self.cfg.device)
			if self.cfg.trainer.aggressive_optimizations:
				batch = to_device(batch, torch.cuda.current_device())
			# engine = engine.to(torch.cuda.current_device())

			while tries >= 0:
				try:
					res = feeder( engines=self, batch=batch, name=name )
					break
				except RuntimeError as e:
					print("Forward", str(e))

					if "out of memory" not in str(e):
						self.save_checkpoint()
						raise e

					# shrink batch size until it's happy
					for k in batch:
						batch[k] = batch[k][:-1]

					if tries <= 0:
						# trigger OOM
						n_ooms += 1
					else:
						# also do GC
						do_gc()
					continue

			all_reduce(n_ooms)
			if n_ooms.item() > 0:
				self.save_checkpoint()
				raise RuntimeError("Out of memory during forward pass!")

			if res is None:
				continue
			
			loss, engine_stats = res

			n_ooms = torch.zeros([], device=self.cfg.device)
			
			if self.cfg.trainer.aggressive_optimizations:
				batch = to_device(batch, 'cpu')

			try:
				engine.backward(loss)
			except RuntimeError as e:
				print("Backwards:", str(e))

				if "out of memory" not in str(e):
					self.save_checkpoint()
					raise e
				
				n_ooms += 1

			all_reduce(n_ooms)
			if n_ooms.item() > 0:
				self.save_checkpoint()
				raise RuntimeError("Out of memory during backwards pass!")

			engine.step()
			torch.cuda.synchronize()
			elapsed_time = time.time() - start_time
			total_elapsed_time += elapsed_time

			stats.update(
				flatten_dict(
					{
						name.split("-")[0]: dict(
							loss=loss.item(),
							lr=engine.get_lr()[0],
							grad_norm=engine.get_global_grad_norm(), # This norm is delayed but global and avoids extra computation
							elapsed_time=elapsed_time,
							engine_step=engine.global_step,
							**engine_stats,
						)
					}
				),
			)
			del loss
			# engine = engine.to('cpu')

		self._update_global_step()
		stats["batch_size"] = len(batch["text"])
		stats["elapsed_time"] = total_elapsed_time
		stats["wall_time"] = time.time()
		stats["global_step"] = self.global_step

		return stats
