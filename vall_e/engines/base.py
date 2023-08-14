from torch import Tensor
from typing import Any, Protocol

Stats = dict[str, float]

class TrainFeeder(Protocol):
	def __call__(
		self, *, engine: "Engine", batch: Any
	) -> None | tuple[Tensor, Stats]:
		...

def default_feeder(engine, batch):
	if isinstance(batch, list):
		engine( *batch )
	elif isinstance(batch, dict):
		engine( **batch )
	else:
		engine( batch )

	losses = engine.gather_attribute("loss")
	loss = torch.stack([*losses.values()]).sum()

	stats = {}
	stats |= {k: v.item() for k, v in losses.items()}

	return loss, stats
	

from ..config import cfg
from ..utils import dispatch_attribute, flatten_dict, gather_attribute, do_gc, to_device
from ..utils.distributed import init_distributed, distributed_initialized

import logging
import time
import torch
import torch.distributed
import os

from torch import Tensor
from torch.distributed import all_reduce
from typing import Any, Protocol

from .base import TrainFeeder

_logger = logging.getLogger(__name__)

if not distributed_initialized() and cfg.trainer.backend == "local":
	init_distributed(torch.distributed.init_process_group)

# A very naive engine implementation using barebones PyTorch
class Engine():
	def __init__(self, *args, **kwargs):
		self.module = kwargs['model'].to(cfg.device).to(cfg.trainer.dtype)
		self.optimizer = kwargs['optimizer'] if 'optimizer' in kwargs else None
		self.lr_scheduler = kwargs['lr_scheduler'] if 'lr_scheduler' in kwargs else None

		self.global_steps = 0
		self.micro_steps = 0
		self.gradient_accumulation_steps = cfg.hyperparameters.gradient_accumulation_steps

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

	def train_batch_size(self):
		return cfg.hyperparameters.batch_size

	def gather_attribute(self, *args, **kwargs):
		return gather_attribute(self.module, *args, **kwargs)

	def dispatch_attribute(self, *args, **kwargs):
		return dispatch_attribute(self.module, *args, **kwargs)

	def save_checkpoint(self, save_dir, tag ):
		save_path = save_dir / tag / "state.pth"
		save_path.parent.mkdir(parents=True, exist_ok=True)
		torch.save({
			"global_step": self.global_step,
			"micro_step": self.micro_step,
			"module": self.module.state_dict(),
			"optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
			"lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
		}, save_path)

		open(save_dir / "latest", 'w').write( tag )

	def load_checkpoint(self, load_dir, tag=None, load_module_strict=True, load_optimizer_states=True, load_lr_scheduler_states=True):
		if tag is None:
			tag_path = load_dir / "latest"
			if not tag_path.exists():
				return
			tag = open(tag_path).read()

		load_path = load_dir / tag / "state.pth"
		if not load_path.exists():
			return

		state = torch.load(load_path)
		self.global_steps = state['global_step']
		self.micro_steps = state['micro_step']
		self.module.load_state_dict(state['module'])

		load_optimizer_states = load_optimizer_states and self.optimizer is not None and 'optimizer' in state
		load_lr_scheduler_states = load_lr_scheduler_states and self.lr_scheduler is not None and 'lr_scheduler' in state
		
		if load_optimizer_states:
			self.optimizer.load_state_dict(state['optimizer'])
		
		if load_lr_scheduler_states:
			self.lr_scheduler.load_state_dict(state['lr_scheduler'])

	def eval(self):
		return self.module.eval()
	
	def train(self):
		return self.module.train()

	def to(self, *args, **kwargs):
		self.module = self.module.to(*args, **kwargs)
		return self.module

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def forward(self, *args, **kwargs):
		return self.module.forward(*args, **kwargs)

	def backward(self, loss):
		return (loss / self.gradient_accumulation_steps).backward()

	def step(self):
		with torch.set_grad_enabled(self.gradient_accumulation_steps > 1):
			self.micro_steps += 1 

			if (self.micro_steps + 1) % max(1, self.gradient_accumulation_steps) == 0:
				self.global_steps += 1 
				self.optimizer.step()
				self.optimizer.zero_grad()

	def get_lr(self):
		lrs = []
		for param_group in self.optimizer.param_groups:
			if 'lr' in param_group:
				lrs.append(param_group['lr'])
		return lrs

	def set_lr(self, lr):
		for param_group in self.optimizer.param_groups:
			if 'lr' in param_group:
				param_group['lr'] = lr

	def get_global_grad_norm(self):
		return 0.0

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

# and now to ignore everything from the above
class Engines(dict[str, Engine]):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.setup()

	def setup(self):
		self._global_step = 0
		self._micro_step = 0

	@property
	def global_step(self):
		return self._global_step
	
	@property
	def micro_step(self):
		return self._micro_step

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
			tag = cfg.trainer.save_tag
		tag = tag.lower()
		if tag[:2] == "it" or tag[:4] == "step":
			tag = f'{self.global_step}'

		cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
		for name, engine in self.items():
			engine.save_checkpoint(cfg.ckpt_dir / name, tag=tag)

	def load_checkpoint(self, tag=None):
		if not tag:
			tag = cfg.trainer.load_tag

		for name, engine in self.items():
			load_dir = cfg.ckpt_dir / name
			engine.load_checkpoint(
				tag=tag,
				load_dir=load_dir,
				load_module_strict=cfg.trainer.strict_loading,
				load_optimizer_states=cfg.trainer.load_states,
				load_lr_scheduler_states=cfg.trainer.load_states,
			)
			if cfg.trainer.restart_step_count:
				engine.global_steps = 0

		# update the LR because for some god awful reason it gets overwritten when loading from a checkpoint but only when it's not using a scheduler
		if cfg.hyperparameters.scheduler_type == "":
			self.set_lr(cfg.hyperparameters.learning_rate)

		self._update_global_step()
		self._update_micro_step()

	def set_lr(self, lr):
		for engine in self.values():
			engine.set_lr(lr)

	def _update_global_step(self):
		for engine in self.values():
			self._global_step = max(self._global_step, engine.global_step)
	
	def _update_micro_step(self):
		for engine in self.values():
			self._micro_step = max(self._micro_step, engine.micro_step)

	def train_batch_size(self):
		batch_size = 0
		for engine in self.values():
			batch_size = max(batch_size, engine.train_batch_size())

	def eval(self):
		for engine in self.values():
			engine.eval()

	def train(self):
		for engine in self.values():
			engine.train()

	def traverse(self):
		stats = {}
		for name, engine in self.items():
			stat = engine.traverse()
			stats.update(flatten_dict({ name.split("-")[0]: stat }))
		return stats

	def step(self, batch, feeder: TrainFeeder = default_feeder):
		total_elapsed_time = 0

		stats: Any = dict()

		if cfg.trainer.gc_mode == 'step':
			do_gc()


		for name, engine in self.items():
			device = engine.device

			if cfg.trainer.gc_mode == 'substep':
				do_gc()

			start_time = time.time()

			tries = 4
			n_ooms = torch.zeros([], device=device)			
			
			batch = to_device(batch, device)

			if not cfg.trainer.check_for_oom:
				res = feeder( engine=engine, batch=batch )
			else:
				while tries >= 0:
					try:
						res = feeder( engine=engine, batch=batch )
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
			engine_stats |= self.gather_attribute("scalar")

			n_ooms = torch.zeros([], device=device)
			
			if cfg.trainer.aggressive_optimizations:
				batch = to_device(batch, 'cpu')

			if not cfg.trainer.check_for_oom:
				engine.backward(loss)
			else:
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
			
			#torch.cuda.synchronize()

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

		self._update_global_step()
		self._update_micro_step()
		stats["batch_size"] = self.train_batch_size() # len(batch["text"])
		stats["elapsed_time"] = total_elapsed_time
		stats["wall_time"] = time.time()
		stats["global_step"] = self.global_step

		return stats
