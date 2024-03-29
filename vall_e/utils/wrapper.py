from contextlib import contextmanager

import torch
import torch.nn.functional as F
from ..config import cfg

Embedding = torch.nn.Embedding
Linear = torch.nn.Linear

# https://github.com/kyegomez/BitNet
if cfg.bitsandbytes.bitnet:
	from bitnet import BitLinear

if cfg.bitsandbytes.enabled:
	import bitsandbytes as bnb

	if cfg.bitsandbytes.linear:

		if cfg.bitsandbytes.bitnet:
			Linear = BitLinear
		else:
			Linear = bnb.nn.Linear8bitLt

	if cfg.bitsandbytes.embedding:
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


if cfg.bitsandbytes.enabled:
	import bitsandbytes as bnb

	Adam = bnb.optim.Adam8bit
	AdamW = bnb.optim.AdamW8bit
	SGD = bnb.optim.SGD8bit
else:
	Adam = torch.optim.Adam
	AdamW = torch.optim.AdamW
	SGD = torch.optim.SGD

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
		with autocasts( input, [torch.int16, torch.int8, torch.uint8], torch.int32 ) as k:
			return func( self, k, *args, **kwargs )
	return wrapper
Embedding.forward = autocast_forward(Embedding.forward)

if cfg.bitsandbytes.injects and cfg.bitsandbytes.enabled:
	torch.nn.Linear = Linear
	torch.nn.Embedding = Embedding

	torch.optim.Adam = Adam
	torch.optim.AdamW = AdamW
	torch.optim.SGD = SGD

# disgusting kludge, but it works (just realized BitNet has its own replacement routine)
def replace_linear( model ):
	device =  next(model.parameters()).device
	linears = [k.split('.') for k, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
	for *parent, k in linears:
		name = '.'.join(parent)

		# copy parameters
		m = getattr( model.get_submodule(name), k )

		in_features = m.in_features
		out_features = m.out_features
		bias = m.bias is not None

		# overwrite
		setattr(
			model.get_submodule(name), k,
			Linear( in_features=in_features, out_features=out_features, bias=bias )
		)

	return model.to(device) # because our now Linear is created on the CPU......

# https://github.com/konstmish/prodigy
try:
	from prodigyopt import Prodigy
except Exception as e:
	pass