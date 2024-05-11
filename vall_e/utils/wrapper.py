from contextlib import contextmanager

import math
import torch
import torch.nn.functional as F

from ..config import cfg

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

# handles generically converting to a specific tensor type and converting back (implemented solely for bfloat16)
@contextmanager
def autocast(input, from_dtype, to_dtype):
	if input.dtype == from_dtype:
		input = input.to(to_dtype)
		yield input
		input = input.to(from_dtype)
	else:
		yield input

@contextmanager
def autocasts(input, from_dtype, to_dtype):
	if input.dtype in from_dtype:
		from_dtype = input.dtype
		input = input.to(to_dtype)
		yield input
		input = input.to(from_dtype)
	else:
		yield input

# handles temporarily upcasting 'index tensors' so torch will stop bitching
def autocast_forward( func ):
	def wrapper( self, input, *args, **kwargs ):
		with autocasts( input, [torch.int16, torch.int8, torch.uint8, torch.float16, torch.bfloat16], torch.int32 ) as k:
			return func( self, k, *args, **kwargs )
	return wrapper
Embedding.forward = autocast_forward(Embedding.forward)

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

# disgusting kludge, but it works (just realized BitNet has its own replacement routine)
# generalizing this would be super sugoi but the there's no catch all for arguments
def replace_linear( model, klass=Linear, target=torch.nn.Linear, verbose=False ):
	bnb = cfg.optimizations.bitsandbytes and cfg.optimizations.linear and not cfg.optimizations.bitnet

	device =  next(model.parameters()).device
	dtype = next(model.parameters()).dtype
	modules = [k.split('.') for k, m in model.named_modules() if isinstance(m, target)]

	for *parent, k in modules:
		name = '.'.join(parent)

		m = getattr( model.get_submodule(name), k )

		if isinstance(m, klass):
			continue

		kwargs = dict(
			in_features = m.in_features,
			out_features = m.out_features,
			bias = m.bias is not None,
		) if not bnb else dict(
			input_features=m.in_features,
			output_features=m.out_features,
			bias=m.bias is not None,
		)

		# overwrite
		setattr(
			model.get_submodule(name), k,
			klass( **kwargs ).to(device=device, dtype=dtype)
		)
		
		if verbose:
			print(f"Replacing {name}.{k} to", klass)

	return model

def replace_embedding( model, klass=Embedding, target=torch.nn.Embedding, verbose=False ):
	device =  next(model.parameters()).device
	dtype = next(model.parameters()).dtype
	modules = [k.split('.') for k, m in model.named_modules() if isinstance(m, target)]

	for *parent, k in modules:
		name = '.'.join(parent)

		m = getattr( model.get_submodule(name), k )

		if isinstance(m, klass):
			continue

		kwargs = dict(
			num_embeddings=m.num_embeddings,
			embedding_dim=m.embedding_dim,
			padding_idx=m.padding_idx,
			max_norm=m.max_norm,
			norm_type=m.norm_type,
			scale_grad_by_freq=m.scale_grad_by_freq,
			sparse=m.sparse,
		)

		# overwrite
		setattr(
			model.get_submodule(name), k,
			klass( **kwargs ).to(device=device, dtype=dtype)
		)
		
		if verbose:
			print(f"Replacing {name}.{k} to", klass)

	return model

# cannot feasibly do default arguments here sad
def replace_attention( model, klass, target, mode="math", verbose=False ):
	device = next(model.parameters()).device
	dtype = next(model.parameters()).dtype
	modules = [k.split('.') for k, m in model.named_modules() if isinstance(m, target)]

	for *parent, k in modules:
		name = '.'.join(parent)

		m = getattr( model.get_submodule(name), k )

		if isinstance(m, klass):
			continue

		kwargs = dict(
			config = m.config,
			layer_idx = m.layer_idx,
			mode = mode,
		)
		# overwrite
		setattr(
			model.get_submodule(name), k,
			klass( **kwargs ).to(device=device, dtype=dtype)
		)
		
		if verbose:
			print(f"Replacing {name}.{k} to", klass)

	return model

# https://github.com/konstmish/prodigy
try:
	from prodigyopt import Prodigy
except Exception as e:
	print('Error while importing Prodigyopt:', str(e))
	pass

# https://github.com/facebookresearch/schedule_free/
try:
	import schedulefree
except Exception as e:
	print('Error while importing Schedule_Free:', str(e))
	pass