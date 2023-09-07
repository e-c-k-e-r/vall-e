from contextlib import contextmanager

import torch
import torch.nn.functional as F
from ..config import cfg

Embedding = torch.nn.Embedding
Linear = torch.nn.Linear

if cfg.bitsandbytes.enabled:
	import bitsandbytes as bnb
	
	if cfg.bitsandbytes.linear:
		Linear = bnb.nn.Linear8bitLt

	if cfg.bitsandbytes.embedding:
		Embedding = bnb.nn.StableEmbedding
		Embedding.forward = lambda self, input: ( self.norm(F.embedding(
			input,
			self.weight,
			self.padding_idx,
			self.max_norm,
			self.norm_type,
			self.scale_grad_by_freq,
			self.sparse,
		)).to(self.weight.dtype) )


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
		"""
		if input.dtype == torch.int16 or input.dtype == torch.int8 or input.dtype == torch.uint8:
			return func( self, input.to(torch.int32), *args, **kwargs )
		return func( self, input, *args, **kwargs )
		"""
	return wrapper
Embedding.forward = autocast_forward(Embedding.forward)

if cfg.bitsandbytes.injects and cfg.bitsandbytes.enabled:
	torch.nn.Linear = Linear
	torch.nn.Embedding = Embedding

	torch.optim.Adam = Adam
	torch.optim.AdamW = AdamW
	torch.optim.SGD = SGD

# https://github.com/konstmish/prodigy
try:
	from prodigyopt import Prodigy
except Exception as e:
	pass