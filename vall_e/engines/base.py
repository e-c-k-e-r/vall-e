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
from ..utils.distributed import init_distributed, distributed_initialized, is_global_leader, world_size, cleanup_distributed
from ..utils.io import torch_save, torch_load
from ..models.lora import freeze_non_lora_weights, lora_get_state_dict, lora_load_state_dict

import logging
import time
import torch
import torch.distributed
import os
import re

from torch import Tensor
from torch.distributed import all_reduce
from typing import Any, Protocol
from functools import cached_property

from .base import TrainFeeder
from ..utils import ml

_logger = logging.getLogger(__name__)

# windows throws an error here
try:
	if not distributed_initialized() and cfg.trainer.backend == "local": # and world_size() > 1:
		init_distributed(torch.distributed.init_process_group)
except Exception as e:
	pass

# A very naive engine implementation using barebones PyTorch
class Engine():
	def __init__(self, *args, **kwargs):
		if 'hyper_config' in kwargs:
			self.hyper_config = kwargs['hyper_config']
			kwargs.pop("hyper_config")

		self.module = kwargs['model'].to(cfg.device).to(torch.float32 if cfg.trainer.amp else cfg.trainer.dtype)
		self.optimizer = kwargs.get('optimizer', None)
		self.lr_scheduler = kwargs.get('lr_scheduler', None)
		self.loss_scaler = torch.cuda.amp.GradScaler() if cfg.trainer.scale_loss else None

		stats = kwargs.get("stats", {})
		if stats is None:
			stats = {}
		
		self.global_steps = stats.get("global_step", 0)
		self.micro_steps = stats.get("micro_step", 0)
		self.global_samples = stats.get("global_samples", 0)
		self.tokens_processed = stats.get("tokens_processed", 0)

		self._frozen_params = set()
		self.current_batch_size = 0
		self._global_grad_norm = None

	def freeze(self, freeze_all=True):
		# set to freeze 
		if self.hyper_config is None or not hasattr(self.hyper_config, "frozen_params"):
			raise Exception("freeze_all=False yet self.hyper_config.frozen_params is None")

		# freeze non-LoRA params if requested
		if not self.hyper_config.frozen_params and not freeze_all and cfg.lora is not None:
			return freeze_non_lora_weights( self.module, embeddings=cfg.lora.embeddings )

		for name, param in self.module.named_parameters():
			if (freeze_all and param.requires_grad) or (not freeze_all and name in self.hyper_config.frozen_params):
				param.requires_grad_(False)
				self._frozen_params.add(param)

	def unfreeze(self):
		for p in self._frozen_params:
			p.requires_grad_(True)
		self._frozen_params.clear()

	@property
	def _training(self):
		if not hasattr(self, "hyper_config"):
			return True
		return self.hyper_config.training

	@property
	def _teacher(self):
		if not hasattr(self, "hyper_config"):
			return False
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

	@property
	def gradient_accumulation_steps(self):
		return cfg.hyperparameters.gradient_accumulation_steps
	
	@property
	def gradient_clipping(self):
		return cfg.hyperparameters.gradient_clipping

	def gather_attribute(self, *args, **kwargs):
		return gather_attribute(self.module, *args, **kwargs)

	def dispatch_attribute(self, *args, **kwargs):
		return dispatch_attribute(self.module, *args, **kwargs)

	def save_checkpoint(self, save_dir, tag ):
		if is_global_leader():
			module = self.module.state_dict()

			if cfg.lora is not None:
				save_dir = cfg.ckpt_dir / cfg.lora.full_name

			save_path = save_dir / tag / f"state.{cfg.weights_format}"
			save_path_optimizer = save_dir / tag / f"optimizer.pth"
			save_path.parent.mkdir(parents=True, exist_ok=True)

			torch_save({
				"module": module,
				"stats": {		
					"global_step": self.global_step,
					"micro_step": self.micro_step,
					"global_samples": self.global_samples,
					"tokens_processed": self.tokens_processed,
				}
			}, save_path)

			torch_save({
				"optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
				"lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
			}, save_path_optimizer )

			open(save_dir / "latest", 'w').write( tag )

		torch.distributed.barrier()

	def load_checkpoint(self, load_dir, tag=None, load_module_strict=True, load_optimizer_states=True, load_lr_scheduler_states=True, load_module_only=False):
		# override to load the lora instead
		if cfg.lora is not None:
			load_dir = cfg.ckpt_dir / cfg.lora.full_name

		if tag is None:
			tag_path = load_dir / "latest"

			if not tag_path.exists():
				return

			tag = open(tag_path).read()

		load_path = load_dir / tag / f"state.{cfg.weights_format}"
		load_path_optimizer = load_dir / tag / f"optimizer.pth"

		if not load_path.exists():
			return
		
		state = torch_load(load_path, device=cfg.device)

		self.global_steps = state['stats']['global_step'] if 'stats' in state else state['global_step']
		self.micro_steps = state['stats']['micro_step'] if 'stats' in state else state['micro_step']
		self.global_samples = state['stats']['global_samples'] if 'stats' in state else state['global_samples']
		self.tokens_processed = state['stats']['tokens_processed'] if 'stats' in state else state['tokens_processed']
		self.module.load_state_dict(state['module'], strict=cfg.trainer.strict_loading)

		if "optimizer" not in state and load_path_optimizer.exists():
			optimizer_state = torch_load(load_path_optimizer, device=cfg.device)
			state["optimizer"] = optimizer_state["optimizer"] if "optimizer" in optimizer_state else None
			state["lr_scheduler"] = optimizer_state["lr_scheduler"] if "lr_scheduler" in optimizer_state else None

		load_optimizer_states = load_optimizer_states and self.optimizer is not None and 'optimizer' in state
		load_lr_scheduler_states = load_lr_scheduler_states and self.lr_scheduler is not None and 'lr_scheduler' in state
		
		if load_optimizer_states:
			self.optimizer.load_state_dict(state['optimizer']) #, device=cfg.device)
		
		if load_lr_scheduler_states:
			self.lr_scheduler.load_state_dict(state['lr_scheduler']) #, device=cfg.device)

		if 'lora' in state and state['lora'] is not None:
			lora_load_state_dict( self.module, state['lora'] )

	def eval(self):
		return self.module.eval()
	
	def train(self):
		return self.module.train()

	def to(self, *args, **kwargs):
		self.module = self.module.to(*args, **kwargs)
		if self.optimizer:
			self.optimizer = self.optimizer.to(*args, **kwargs)

		return self

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	@cached_property
	def device(self):
		return next(self.module.parameters()).device

	def forward(self, *args, **kwargs):
		return self.module.forward(*args, **kwargs)

	def backward(self, loss):
		if self.loss_scaler is not None:
			return self.loss_scaler.scale(loss / self.gradient_accumulation_steps).backward()
		return (loss / self.gradient_accumulation_steps).backward()


	def step(self):
		with torch.set_grad_enabled(self.gradient_accumulation_steps > 1):
			self.micro_steps += 1 
			self.global_samples += self.batch_size

			if (self.micro_steps + 1) % max(1, self.gradient_accumulation_steps) == 0:
				self._global_grad_norm = torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.gradient_clipping)

				self.global_steps += 1 
				if self.loss_scaler is not None:
					self.loss_scaler.step(self.optimizer)
					self.loss_scaler.update()
				else:
					self.optimizer.step()
				
				if self.lr_scheduler is not None:
					self.lr_scheduler.step()
				
				self.optimizer.zero_grad()
	
	# doesn't actually work
	def _get_grad_norm(self):
		t = [ param.grad.detach().flatten() for param in self.module.parameters() if param.grad is not None ]
		self._global_grad_norm = torch.cat(t).norm().item() if len(t) else None

	def get_lr(self):
		lrs = []
		for param_group in self.optimizer.param_groups:
			if 'd_coeff' in param_group:
				lrs.append(param_group['d_coeff'])
			elif 'lr' in param_group:
				lrs.append(param_group['lr'])
		return lrs

	def set_lr(self, lr):
		for param_group in self.optimizer.param_groups:
			if 'd_coeff' in param_group:
				param_group['d_coeff'] = lr
			elif 'lr' in param_group:
				param_group['lr'] = lr

	def get_loss_scale(self):
		if not hasattr(self, "loss_scaler") or self.loss_scaler is None:
			return 1

		return self.loss_scaler.get_scale()

	def set_loss_scale(self, value):
		if not hasattr(self, "loss_scaler") or self.loss_scaler is None:
			return
		
		"""
		self.optimizer.loss_scale = value
		"""

	def get_global_grad_norm(self):
		return self._global_grad_norm

	def traverse(self, *args, **kwargs):
		with ml.autocast():
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
		self._batch_size = 0
		self._global_samples = 0

	@property
	def global_step(self):
		return self._global_step
	
	@property
	def micro_step(self):
		return self._micro_step

	@property
	def batch_size(self):
		return self._batch_size

	@property
	def global_samples(self):
		return self._global_samples

	def gather_attribute(self, *args, **kwargs):
		ret = {}
		for engine in self.values():
			ret |= engine.gather_attribute(*args, **kwargs)
		return ret

	def dispatch_attribute(self, *args, **kwargs):
		for engine in self.values():
			engine.dispatch_attribute(*args, **kwargs)

	def export(self, userdata={}, callback=None, dtype=None, format=None):
		if not format:
			format = cfg.weights_format
		format = format.lower()

		if dtype is None:
			dtype = cfg.trainer.dtype

		for name, engine in self.items():
			module = engine.module.state_dict()
			lora = None
			save_path = cfg.ckpt_dir / name / f"{cfg.weights_name}.{format}"
			config = engine.module.config if hasattr(engine.module, "config") else engine.hyper_config

			# safety
			for k, v in module.items():
				module[k] = v.to(dtype)

			if cfg.lora is not None:				
				lora, module = lora_get_state_dict( module, split = True )
				save_path = cfg.ckpt_dir / cfg.lora.full_name / f"{cfg.weights_name}.{format}"

			config_dict = dict(**config.__dict__)
			config_dict |= {"experimental": config.experimental.__dict__}

			state_dict = {
				'module': module,
				'lora': lora,
				"stats": {
					"global_step": engine.global_step,
					"micro_step": engine.micro_step,
					"global_samples": engine.global_samples,
					"tokens_processed": engine.tokens_processed,
				},
				"userdata": userdata,
				"config": config_dict
			}

			if lora is None:
				del state_dict['lora']

			if callback:
				state_dict = callback( state_dict, config = engine.hyper_config, save_path = save_path )

			torch_save(state_dict, save_path)
			_logger.info(f"Exported {name} to {save_path}")

	def save_checkpoint(self, tag=None):
		if not tag:
			tag = cfg.trainer.save_tag
		tag = tag.lower()
		if tag[:2] == "it" or tag[:4] == "step":
			tag = f'{self.global_step}'

		cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
		for name, engine in self.items():
			if not engine._training:
				continue

			save_dir = cfg.ckpt_dir / name
			if cfg.lora is not None:
				save_dir = cfg.ckpt_dir / cfg.lora.full_name

			engine.save_checkpoint(save_dir, tag=tag)

			"""
			try:
				engine.save_checkpoint(save_dir, tag=tag)
			except Exception as e:
				_logger.warning(f'Failed to save checkpoint for engine {name}: {str(e)}')
			"""

			# might be better to prune before saving for safety, but [:0] returns an empty list, but I could do [:-cfg.trainer.keep_last_checkpoints - 1 if cfg.trainer.keep_last_checkpoints > 1 else None]
			if cfg.trainer.keep_last_checkpoints > 0 and is_global_leader():
				checkpoints = [ d for d in list(save_dir.glob("*")) if d.is_dir() ]
				checkpoints.sort(key=lambda x: x.stat().st_mtime)
				checkpoints = checkpoints[:-cfg.trainer.keep_last_checkpoints]
				for d in checkpoints:
					if not d.is_dir() or not d.exists():									
						continue
					_logger.info(f"Removing {d}")
					for p in d.iterdir():
						p.unlink()
					d.rmdir()

	def load_checkpoint(self, tag=None, training=True):
		if not tag:
			tag = cfg.trainer.load_tag

		for name, engine in self.items():
			load_dir = cfg.ckpt_dir / name

			engine.load_checkpoint(
				tag=tag,
				load_dir=load_dir,
				load_module_strict=cfg.trainer.strict_loading,
				load_optimizer_states=False if cfg.trainer.load_module_only or not training else cfg.trainer.load_states,
				load_lr_scheduler_states=False if cfg.trainer.load_module_only or not training else cfg.trainer.load_states,
				load_module_only=cfg.trainer.load_module_only,
			)
			if cfg.trainer.restart_step_count:
				engine.global_steps = 0
				engine.mocro_step = 0
				engine.global_samples = 0
				engine.tokens_processed = 0

		# update the LR because for some god awful reason it gets overwritten when loading from a checkpoint but only when it's not using a scheduler
		if cfg.hyperparameters.scheduler == "":
			self.set_lr(cfg.hyperparameters.learning_rate)

		self._update()

	def set_lr(self, lr):
		for engine in self.values():
			if not engine._training:
				continue
			engine.set_lr(lr)

	def set_loss_scale(self, lr):
		for engine in self.values():
			if not engine._training:
				continue
			engine.set_loss_scale(lr)

	def _update(self):
		for engine in self.values():
			self._global_step = max(self._global_step, engine.global_step)
			self._micro_step = max(self._micro_step, engine.micro_step)
			self._batch_size = max(self._batch_size, engine.batch_size)
			self._global_samples = max(self._global_samples, engine.global_samples)

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

	def quit(self):
		for name, engine in self.items():
			if engine.wandb is not None:
				engine.wandb.finish()
		
		cleanup_distributed()

	def step(self, batch, feeder: TrainFeeder = default_feeder):
		total_elapsed_time = 0

		stats: Any = dict()

		if cfg.trainer.gc_mode == 'step':
			do_gc()

		# preiterate to get teacher
		teacher = None
		for name, engine in self.items():
			if not engine._teacher:
				continue
			teacher = engine.module
			break

		for name, engine in self.items():
			# only models that we're training
			if not engine._training or engine._teacher:
				continue

			device = engine.device

			if cfg.trainer.gc_mode == 'substep':
				do_gc()

			start_time = time.time()

			batch = to_device(batch, device)

			if not cfg.trainer.check_for_oom:
				res = feeder( engine=engine, batch=batch, teacher=teacher )
			else:
				forward_ooms = torch.zeros([], device=device)
				try:
					res = feeder( engine=engine, batch=batch, teacher=teacher )
				except RuntimeError as e:
					_logger.error(f"Forward: {str(e)}")

					if "out of memory" not in str(e):
						self.save_checkpoint()
						raise e

					forward_ooms += 1

				if world_size() > 1:
					all_reduce(forward_ooms)

				if forward_ooms.item() > 0:
					continue
					"""
					self.save_checkpoint()
					raise RuntimeError("Out of memory during forward pass!")
					"""

			# this causes problems in distributed training
			# it's probably required to do all_reduce for nan checks
			"""
			# no results are returned when a nan is encountered, so catch it here too
			if res is None:
				engine.max_nan_losses = engine.max_nan_losses - 1
				if engine.max_nan_losses < 0:
					raise RuntimeError("Too many NaN losses detected.")
				continue
			"""
			
			loss, engine_stats = res
			engine_stats |= self.gather_attribute("scalar")

			if not cfg.trainer.check_for_oom:
				engine.backward(loss)
			else:
				backward_ooms = torch.zeros([], device=device)
				try:
					engine.backward(loss)
				except RuntimeError as e:
					_logger.error(f"Backwards: {str(e)}")

					if "out of memory" not in str(e):
						self.save_checkpoint()
						raise e
					
					backward_ooms += 1

				if world_size() > 1:
					all_reduce(backward_ooms)

				if backward_ooms.item() > 0:
					self.save_checkpoint()
					raise RuntimeError("Out of memory during backwards pass!")

			engine.step()
			
			#torch.cuda.synchronize()

			elapsed_time = time.time() - start_time
			total_elapsed_time += elapsed_time
			grad_norm = engine.get_global_grad_norm()
			loss_scale = engine.get_loss_scale()

			if cfg.trainer.deepspeed.max_loss_scale > 0 and loss_scale > cfg.trainer.deepspeed.max_loss_scale:
				_logger.warning(f'Loss scale ({loss_scale}) exceeds max_loss_scale ({cfg.trainer.deepspeed.max_loss_scale}), capping...')
				engine.set_loss_scale(cfg.trainer.deepspeed.max_loss_scale)

			# scale the grad norm to normal, if not using ZeRO because ZeRO does this already
			if grad_norm is not None and not cfg.trainer.deepspeed.zero_optimization_level:
				grad_norm /= loss_scale

			model_stats = dict(
				**engine_stats,
				grad_norm=grad_norm.item() if isinstance( grad_norm, torch.Tensor ) else grad_norm,
				loss_scale=loss_scale if loss_scale != 1 else None,
			)
		
			if engine.wandb is not None:
				engine.wandb.log(model_stats, step=engine.global_step)

			filtered_keys = [ k for k in model_stats.keys() if "[" in k ]
			filtered_values = {}
			for k in filtered_keys:
				v = model_stats[k]
				del model_stats[k]

				nk = re.sub(r"\[\d+\]", "", k)
				
				if nk not in filtered_values:
					filtered_values[nk] = []
				
				filtered_values[nk].append( v )

			for k, v in filtered_values.items():
				model_stats[k] = sum(v) / len(v)

			model_stats = model_stats | dict(
				lr=engine.get_lr()[0],
				elapsed_time=elapsed_time,
				engine_step=engine.global_step,
				samples_processed=engine.global_samples,
				tokens_processed=engine.tokens_processed,
			)

			key_name = name
			if cfg.lora is not None:			
				key_name = cfg.lora.full_name

			if len(self) == 1:
				stats.update(flatten_dict(model_stats))
			else:
				stats.update(flatten_dict({key_name.split("-")[0]: model_stats}))

		self._update()

		if len(self.keys()) > 1:
			stats["elapsed_time"] = total_elapsed_time
		
		stats["it"] = self.global_step

		return stats
