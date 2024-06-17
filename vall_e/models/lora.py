# Adapted from https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
from functools import partial
import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from torch import Tensor, nn

import math
from typing import Optional, List

# to-do: set cfg to decide
USE_PARAMETRIZATION = True

# LoRA Linear for replacement
# Pros: simple, just needs to reuse the replace_linear and copy weights
# Cons: does not work with other Linears (bnb, bitnet, te's fp8, etc), cannot apply multiple LoRAs (although for audio why would you)
class Linear(nn.Linear):
	def __init__(
		self, 
		
		in_features: int, 
		out_features: int,
		bias: bool = True,

		rank: int = 4, 
		alpha: int = 1, 
		
		dropout: float = 0.1,
		merge_weights: bool = True,
		**kwargs,
	):
		super().__init__(in_features=in_features, out_features=out_features, bias=bias, **kwargs)

		self.rank = rank
		self.alpha = alpha
		self.dropout = nn.Dropout(p=dropout) if dropout > 0 else lambda x: x
		self.merge_weights = merge_weights
		self.merged = False

		self.lora_B = nn.Parameter( self.weight.new_zeros( (out_features, rank) ) )
		self.lora_A = nn.Parameter( self.weight.new_zeros( (rank, in_features) ) )
		self.scaling = self.alpha / self.rank
		
		self.weight.requires_grad = False

		self.reset_parameters()

	def reset_parameters(self):
		super().reset_parameters()
		# super silly but necessary because nn.Linear's constructor calls this
		if hasattr(self, 'lora_A'):
			nn.init.kaiming_uniform_( self.lora_A, a=math.sqrt(5) )
			nn.init.zeros_( self.lora_B )

	def train(self, mode: bool = True):
		super().train(mode)

		# training, separate lora from base weights
		if mode and self.merge_weights and self.merged:
			self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
			self.merged = False

		# not training, merge lora to base weights
		if not mode and self.merge_weights and not self.merged:
			self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
			self.merged = True   

	def forward(self, x: torch.Tensor):
		if not self.merged:
			result = F.linear(x, self.weight, bias=self.bias)			
			result += (self.dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
			return result

		return F.linear(x, self.weight, bias=self.bias)

	@classmethod
	def from_linear( cls, layer, device = None, dtype = None, **kwargs ):
		if device is None:
			device = layer.weight.device
		if dtype is None:
			dtype = layer.weight.dtype
		return cls( in_features = layer.in_features, out_features = layer.out_features, bias = layer.bias is not None, **kwargs ).to(device=device, dtype=dtype)

# Uses parametrization to inject LoRA weights
# Pros: should work with any Linears
# Cons: TBD
class ParameterizedLinear(nn.Module):
	def __init__(
		self, 
		
		in_features: int, 
		out_features: int,
		bias: bool = True,

		rank: int = 4, 
		alpha: int = 1, 
		
		dropout: float = 0.1,

		device = None,
		dtype = None
	):
		super().__init__()
		self.rank = rank
		self.alpha = alpha
		self.dropout = nn.Dropout(p=dropout) if dropout > 0 else lambda x: x

		self.lora_B = nn.Parameter( torch.zeros( (out_features, rank) ) ).to( device=device, dtype=dtype )
		self.lora_A = nn.Parameter( torch.zeros( (rank, in_features) ) ).to( device=device, dtype=dtype )
		self.scaling = self.alpha / self.rank
		self.enabled = True
		
		self.reset_parameters()

	def reset_parameters(self):
		nn.init.kaiming_uniform_( self.lora_A, a=math.sqrt(5) )
		nn.init.zeros_( self.lora_B ) 

	def forward(self, x: torch.Tensor):
		if self.enabled:
			return x + torch.matmul(self.lora_B, self.dropout(self.lora_A)).view(x.shape) * self.scaling

		return x

	@classmethod
	def from_linear( cls, layer, device = None, dtype = None, **kwargs ):
		if device is None:
			device = layer.weight.device
		if dtype is None:
			dtype = layer.weight.dtype
		# swap because we're feeding the output as our input
		# M$'s LoRA class arranges things to where this isn't necessary
		return cls( in_features = layer.out_features, out_features = layer.in_features, bias = layer.bias is not None, **kwargs ).to(device=device, dtype=dtype)

def passes_policy( policy, name ):
	if policy is None:
		return True

	if "exclude" in policy:
		for term in policy["exclude"]:
			if term in name:
				return False

	if "include" in policy:
		for term in policy["include"]:
			if term in name:
				return True

	return False


def apply_lora( model, register = True, merge = False, policy = None, **kwargs ):
	device =  next(model.parameters()).device
	dtype = next(model.parameters()).dtype

	klass = Linear
	target = nn.Linear

	device =  next(model.parameters()).device
	dtype = next(model.parameters()).dtype
	modules = [ k.split('.') for k, m in model.named_modules() if isinstance(m, target) and not isinstance(m, klass) and passes_policy( policy, k ) ]

	for *parent, k in modules:
		name = '.'.join(parent)
		layer = getattr( model.get_submodule(name), k )

		if USE_PARAMETRIZATION:
			parametrize.register_parametrization( layer, "weight", ParameterizedLinear.from_linear( layer, device=device, dtype=dtype, **kwargs ) )
			# parametrize.remove_parametrizations( layer, "weight", leave_parametrized=merge )
		else:
			setattr( model.get_submodule(name), k, Linear.from_linear( layer, device=device, dtype=dtype, **kwargs ) )

	return model

def freeze_non_lora_weights( model ):
	for name, param in model.named_parameters():
		param.requires_grad_('lora_' in name)
	return model

def lora_get_state_dict( state_dict, split = True ):
	lora = { name: param for name, param in state_dict.items() if "lora_" in name }
	if not split:
		return lora

	return lora, { name: param for name, param in state_dict.items() if "lora_" not in name }

def lora_load_state_dict( model, state_dict ):
	return model.load_state_dict( state_dict, strict = False )