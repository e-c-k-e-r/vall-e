from contextlib import contextmanager

import math
import logging

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

from ..config import cfg

_logger = logging.getLogger(__name__)

Embedding = torch.nn.Embedding
Linear = torch.nn.Linear

Adam = torch.optim.Adam
AdamW = torch.optim.AdamW
SGD = torch.optim.SGD
Adagrad = torch.optim.Adagrad
Adafactor = torch.optim.Adafactor

OneCycleLR = torch.optim.lr_scheduler.OneCycleLR
CosineAnnealingLR = torch.optim.lr_scheduler.CosineAnnealingLR
LambdaLR = torch.optim.lr_scheduler.LambdaLR

# implements Noam scheduling
# it's cringe
class NoamLR(_LRScheduler):
	def __init__(self, optimizer, warmup_steps, d_model=1024, last_epoch=-1):
		self.base_factor = d_model ** (-0.5)
		self.warmup_steps = warmup_steps
	
		super().__init__(optimizer, last_epoch)

	def get_lr(self):
		step = max(1, self.last_epoch)
		scale = self.base_factor * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))

		return [base_lr * scale for base_lr in self.base_lrs]

# gradually warms up LR then holds or decays
class WarmupLR(_LRScheduler):
	def __init__(self, optimizer, warmup_steps, decay_factor=0.0, last_epoch=-1):
		self.warmup_steps = warmup_steps
		self.decay_factor = decay_factor

		super().__init__(optimizer, last_epoch)

	def get_lr(self):
		step = self.last_epoch + 1
		scale = 1
		if step < self.warmup_steps:
			scale = float(step) / float(max(1, self.warmup_steps))
		elif self.decay_factor != 0:
			scale = (1.0 - self.decay_factor) ** (step - self.warmup_steps)

		return [base_lr * scale for base_lr in self.base_lrs]

# https://github.com/kyegomez/BitNet
if cfg.optimizations.bitnet:
	from bitnet import BitLinear

if cfg.optimizations.bitsandbytes:
	import bitsandbytes as bnb

	if cfg.optimizations.linear:

		if cfg.optimizations.bitnet:
			Linear = BitLinear
		else:
			Linear = bnb.nn.Linear8bitLt

	if cfg.optimizations.embedding:
		Embedding = bnb.nn.StableEmbedding
		"""
		Embedding.forward = lambda self, input: ( self.norm(F.embedding(
			input,
			self.weight,
			self.padding_idx,
			self.max_norm,
			self.norm_type,
			self.scale_grad_by_freq,
			self.sparse,
		)).to(self.weight.dtype) )
		"""

	if cfg.optimizations.optimizers:
		Adam = bnb.optim.Adam8bit
		AdamW = bnb.optim.AdamW8bit
		SGD = bnb.optim.SGD8bit
		Adagrad = bnb.optim.Adagrad8bit

elif cfg.optimizations.dadaptation:
	import dadaptation

	if cfg.optimizations.optimizers:
		Adam = dadaptation.DAdaptAdam
		AdamW = dadaptation.DAdaptAdam
		SGD = dadaptation.DAdaptSGD
		AdaGrad = dadaptation.DAdaptAdaGrad

if cfg.optimizations.fp8:
	import transformer_engine.pytorch as te

	Linear = te.Linear

	@contextmanager
	def autocast():
		yield te.fp8_autocast(enabled=True)
else:
	@contextmanager
	def autocast():
		yield torch.autocast("cuda", dtype=cfg.trainer.dtype, enabled=cfg.trainer.amp)

if cfg.optimizations.injects:
	if cfg.optimizations.linear:
		torch.nn.Linear = Linear
	
	if cfg.optimizations.embedding:
		torch.nn.Embedding = Embedding

	if cfg.optimizations.optimizers:
		torch.optim.Adam = Adam
		torch.optim.AdamW = AdamW
		torch.optim.SGD = SGD

if cfg.optimizations.unsloth:
	try:
		from .ext.unsloth import apply_unsloth_offloaded_gradient_checkpoint_monkey_patch
		#apply_unsloth_offloaded_gradient_checkpoint_monkey_patch()
	except Exception as e:
		_logger.warning(f'Error while importing Unsloth: {str(e)}')
		pass

class Optimizers(torch.optim.Optimizer):
	def __init__(self, opts):
		self.opts = opts

	def step(self, *args, **kwargs):
		for opt in self.opts:
			opt.step(*args, **kwargs)
	
	def zero_grad(self, *args, **kwargs):
		for opt in self.opts:
			opt.zero_grad(*args, **kwargs)

	@property
	def param_groups(self):
		l = []
		for opt in self.opts:
			l += opt.param_groups
		return l

	def state_dict(self):
		states = []
		for i, opt in enumerate( self.opts ):
			states.append( opt.state_dict() )
		
		return states

	def load_state_dict(self, state_dict):		
		for opt, state in zip( self.opts, state_dict ):
			opt.load_state_dict( state )

try:
	from .ext.apollo import Apollo
except Exception as e:
	_logger.warning(f'Error while importing APOLLO: {str(e)}')
	pass

try:
	from .ext.muon import Muon
except Exception as e:
	_logger.warning(f'Error while importing Muon: {str(e)}')
	pass

# https://github.com/konstmish/prodigy
try:
	from prodigyopt import Prodigy
except Exception as e:
	_logger.warning(f'Error while importing Prodigyopt: {str(e)}')
	pass

# https://github.com/facebookresearch/schedule_free/
try:
	import schedulefree
except Exception as e:
	_logger.warning(f'Error while importing Schedule_Free: {str(e)}')
	pass

# backwards compat
from .utils import (
	autocast_forward,
	replace_linear as replace_linear_old,
	replace_embedding as replace_embedding_old,
	replace_attention,
	resize_weight,
	offload_model,
)

# wrapped here so we can maintain default args
def replace_linear( model, klass=Linear, target=torch.nn.Linear, verbose=False ):
	return replace_linear_old( model, klass, target, verbose )
def replace_embedding( model, klass=Embedding, target=torch.nn.Embedding, verbose=False ):
	return replace_embedding_old( model, klass, target, verbose )

Embedding.forward = autocast_forward(Embedding.forward)

AVAILABLE_COMPILE_BACKENDS = []

try:
	AVAILABLE_COMPILE_BACKENDS += torch._dynamo.list_backends()
except Exception as e:
	pass

def compile_model(model, backend="auto"):
	if not backend or backend == "auto":
		backend = AVAILABLE_COMPILE_BACKENDS[0]

	if backend not in AVAILABLE_COMPILE_BACKENDS:
		return torch.compile(model)

	return torch.compile(model, backend=backend)


if cfg.optimizations.tensorrt:
	try:
		import torch_tensorrt
		AVAILABLE_COMPILE_BACKENDS.append("tensorrt")
	except Exception as e:
		_logger.warning(f'Error while importing TensorRT: {str(e)}')
		pass