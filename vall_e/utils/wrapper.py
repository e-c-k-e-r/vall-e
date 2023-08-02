# to-do: re-introduce bitsandbytes support

from contextlib import contextmanager

import torch
import torch.nn.functional as F

Embedding = torch.nn.Embedding
Linear = torch.nn.Linear

"""
if cfg.bitsandbytes:
	import bitsandbytes as bnb
	
	if cfg.bitsandbytes_linear:
		Linear = bnb.nn.Linear8bitLt

	if cfg.bitsandbytes_embedding:
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
"""

Adam = torch.optim.Adam
AdamW = torch.optim.AdamW

"""
if cfg.bitsandbytes:
	import bitsandbytes as bnb

	Adam = bnb.optim.Adam
	AdamW = bnb.optim.AdamW
"""

# handles temporarily upcasting 'index tensors' so torch will stop bitching
def autocast_forward( func ):
	def wrapper( self, input, *args, **kwargs ):
		if input.dtype == torch.int16 or input.dtype == torch.int8 or input.dtype == torch.uint8:
			input = input.to(torch.int32)

		return func( self, input, *args, **kwargs )
	return wrapper
Embedding.forward = autocast_forward(Embedding.forward)

# handles generically converting to a specific tensor type and converting back (implemented solely for bfloat16)
@contextmanager
def autocast(input, from_dtype, to_dtype):
	if input.dtype == from_dtype:
		input = input.to(to_dtype)
		yield input
		input = input.to(from_dtype)
	else:
		yield input