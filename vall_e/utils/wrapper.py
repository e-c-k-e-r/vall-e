from contextlib import contextmanager

import math
import torch
import torch.nn.functional as F
import logging

from ..config import cfg

_logger = logging.getLogger(__name__)

Embedding = torch.nn.Embedding
Linear = torch.nn.Linear

Adam = torch.optim.Adam
AdamW = torch.optim.AdamW
SGD = torch.optim.SGD
Adagrad = torch.optim.Adagrad

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
		Embedding = bnb.nn.modules.Embedding
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

AVAILABLE_COMPILE_BACKENDS = []

try:
	AVAILABLE_COMPILE_BACKENDS += torch._dynamo.list_backends()
except Exception as e:
	pass


if cfg.optimizations.tensorrt:
	try:
		import torch_tensorrt
		AVAILABLE_COMPILE_BACKENDS.append("tensorrt")
	except Exception as e:
		_logger.warning(f'Error while importing TensorRT: {str(e)}')
		pass

if cfg.optimizations.unsloth:
	try:
		from .ext.unsloth import apply_unsloth_offloaded_gradient_checkpoint_monkey_patch
		#apply_unsloth_offloaded_gradient_checkpoint_monkey_patch()
	except Exception as e:
		_logger.warning(f'Error while importing Unsloth: {str(e)}')
		pass

try:
	from .ext.apollo import Apollo
except Exception as e:
	_logger.warning(f'Error while importing APOLLO: {str(e)}')
	pass

def compile_model(model, backend="auto"):
	if not backend or backend == "auto":
		backend = AVAILABLE_COMPILE_BACKENDS[0]

	if backend not in AVAILABLE_COMPILE_BACKENDS:
		return torch.compile(model)

	return torch.compile(model, backend=backend)

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