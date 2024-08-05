"""
Core model for handling all VALL-E tasks.
This should handle all the "low" level things such as:
* parsing inputs to sequences
* converting sequences to embeddings
* forward pass
* processing loss and returning logits

Additional functionality (preparing inputs, generating full audio) should be delegated to classes that inheret the base model
"""

import math
import torch
import torch.nn.functional as F
import random
import numpy as np
import re

from typing import Literal, overload, Optional, Tuple
from functools import partial
from einops import rearrange

from torch import Tensor, einsum, nn
from torch.nn import Embedding
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy, MulticlassPrecision

from .arch import *
from ..utils import wrapper as ml
from ..samplers import *

from ..emb.qnt import encode_as_embedding

# yuck, kind of needed
from ..data import get_task_symmap

"""
from ..utils.pattern import DelayedPatternProvider, VALLEPattern
"""

def _create_mask(l, device):
	"""1 is valid region and 0 is invalid."""
	seq = torch.arange(max(l), device=device).unsqueeze(0)  # (1 t)
	stop = torch.tensor(l, device=device).unsqueeze(1)  # (b 1)
	return (seq < stop).float()  # (b t)

def _join(x: tuple[Tensor], sep: Tensor):
	"""
	Args:
		x: (k t d)
		sep: (d)
	"""
	ret = x[0]
	for i in range(1, len(x)):
		ret = torch.cat((ret, sep[None], x[i]), dim=0)
	return ret

def list_to_tensor(x_list: list[Tensor], pattern="t b c -> b t c"):
	"""
	Args:
		x_list: [(t d)]
	Returns:
		x: (? ? ?)
		m: (? ? ?), same as x
	"""
	l = list(map(len, x_list))
	x = rearrange(pad_sequence(x_list), pattern)
	m = _create_mask(l, x_list[0].device)
	m = m.t().unsqueeze(-1)  # (t b 1)
	m = rearrange(m, pattern)
	m = m.to(x)
	return x, m

def _interleave_sequence_reshape( input: list[torch.Tensor], dim=-1 ):
	shape = (input[0].shape[0] * len(input), input[0].shape[dim] )
	return torch.concat( [ i.t() for i in input ] ).t().reshape( shape )

def _interleave_sequence_flatten( input: list[torch.Tensor] ):
	return torch.concat( [ i.t() for i in input ] ).t().flatten()

# automagically parses a batch-list and returns it as a list
"""
class Embedding(nn.Embedding):
	def forward(self, x_list: list[Tensor]) -> list[Tensor]:
		if len(x_list) == 0:
			return []
		return super().forward(torch.cat(x_list)).split([*map(len, x_list)])
"""

# Deprecated implementation
class MultiEmbedding(nn.Module):
	def __init__(self, max_n_levels, n_tokens, token_dim, monolithic=False):
		super().__init__()
		self.monolithic = monolithic
		self.max_n_levels = max_n_levels
		self.n_tokens = n_tokens
		self.weight = nn.Parameter(torch.randn(max_n_levels, n_tokens, token_dim))

	# to-do: select quant level from given quant_levels tensor if given (i.e. through the resps_emb)
	# I imagine this is an oversight in the NAR.
	def forward(self, x_list: list[Tensor], quant_level: int | list[int] | Tensor | None = None) -> list[Tensor]:
		if len(x_list) == 0:
			return []

		# this "strategy" will reserve the weight[0] for te AR and weight[1:] for the NAR
		# the NAR cannot share RVQ-bin level 0 with the AR for the resps_emb
		if self.monolithic:
			w = self.weight[:1] if quant_level is None or quant_level == 0 else self.weight[1:]
		else:
			w = self.weight

		padded_x_list = []

		for i, xi in enumerate(x_list):
			xi = F.one_hot(xi.to(torch.int64), num_classes=self.n_tokens)  # t l' k
			wi = w.shape[0] - xi.shape[1]
			xi = F.pad(xi, (0, 0, 0, wi))  # t l k
			padded_x_list.append(xi.to(w))

		x = torch.cat(padded_x_list)  # n l k
		x = einsum("l k d, n l k -> n d", w, x)

		x_list = x.split([*map(len, x_list)])

		return x_list

# Embedding that sums each RVQ-bin level within a given input acoustic prompt
# _Old, to preserve compat with previous models.
class AudioEmbedding_Old(nn.Module):
	def __init__(
		self,
		l_tokens: int, # list of number of tokens (needed because AR resps includes stop token)
		token_dim: int, # dimensionality of the embedding
		levels: int | None = None, # number of RVQ-bins (I don't remember the specifics)
	):
		super().__init__()
		# array of embeddings
		#   proms are [0, resp_levels]
		#   resp are split to where [0] is for the AR, and [1:] are reserved for NAR
		self.embeddings = nn.ModuleList([nn.Embedding(n_tokens, token_dim) for n_tokens in l_tokens])
		# weight influencer for the influence for each level (desu this should be really useless because the weights in the embedding themselves should factor this)
		self.weight = nn.ParameterList([nn.Parameter( torch.tensor([1]) ) for i in range(levels)]) if levels is not None else None

	def forward(self, xi: Tensor, quant_level: Tensor | None = None ) -> Tensor:
		# prom
		if quant_level is None and xi.shape[-1] > 1:
			x = sum( [ self.embeddings[k]( xi[:, k] ) * (self.weight[k] if self.weight is not None else 1) for k in range(xi.shape[-1]) ] )
		# prom / AR resp
		elif quant_level is None or quant_level == 0:
			x = self.embeddings[0]( xi if xi.dim() == 1 else xi[:, 0] )
		# NAR resp
		else:
			x = sum( [ self.embeddings[k+1]( xi[:, k] ) * (self.weight[k+1] if self.weight is not None else 1) for k in range(xi.shape[-1]) ] )

		return x

# Embedding that sums each RVQ-bin level within a given input acoustic prompt
# Mostly to handle some oversights and errors during testing
class AudioEmbedding(nn.Module):
	def __init__(
		self,
		l_tokens: list[int], # list of number of tokens (needed because AR resps includes stop token)
		token_dim: int, # dimensionality of the embedding
		sums: bool = True, # whether to sum all previous layers of embeddings to factor in other RVQ bin levels (I do not know which way is better)
		external_mode: str | None = None, # "exclusive" | "inclusive", whether to include the original audio backend's embeddings

		capabilities: list[str] | None = None, # helper shit
	):
		super().__init__()
		# array of embeddings
		#   proms are [0, resp_levels]
		#   resp are split to where [0] is for the AR, and [1:] are reserved for NAR
		#	 + resps cannot share the AR and NAR embeddings, since they do encode whether to predict the same level but in the next token or predict in place but the next level
		self.embeddings = nn.ModuleList([nn.Embedding(n_tokens, token_dim) for n_tokens in l_tokens])
		# further experimentation is needed to see if this actually is useful
		self.sums = sums

		self.external_mode = external_mode
		self.capabilities = capabilities

		# set initial weights to zero
		if self.external_mode == "inclusive":
			for i, embedding in enumerate(self.embeddings):
				embedding.weight = torch.nn.Parameter(torch.zeros( embedding.weight.shape ))

	def external_embeddings(self, input: Tensor, quant_level: int | None = None ) -> Tensor:
		if quant_level is None:
			quant_level = 0 if input.dim() == 1 else input.shape[-1] - 1

		# for AR, trim any stop tokens
		has_stop_token = False
		
		# this block apparently doesn't work
		"""
		if quant_level == 0:
			stop_token = self.embeddings[0].weight.shape[0] - 1
			stop_token_indices = (input == stop_token).nonzero()
			has_stop_token = len(stop_token_indices) > 0
		
		if has_stop_token:
			input = input[:stop_token_indices.min().item()]
		"""
		has_stop_token = False

		if quant_level == 0:
			stop_token = self.embeddings[0].weight.shape[0] - 1
			has_stop_token = input[-1] == stop_token

		if has_stop_token:
			input = input[:-1]

		# get external embedding
		embedding = encode_as_embedding( input, quant_level, sums=self.sums ).to(device=input.device, dtype=self.embeddings[quant_level].weight.dtype)
		# resize if necessary (in case the external embeddings do not match our model dim)
		embedding = ml.resize_weight( embedding, self.embeddings[quant_level].weight.shape[-1], dim=-1, random=False )

		# reintroduce stop token
		if has_stop_token:
			stop_token = self.internal_forward( torch.tensor([stop_token]).to(device=input.device, dtype=torch.int16), 0 )
			embedding = torch.concat( [ embedding, stop_token ] )

		return embedding

	def internal_forward(self, xi: Tensor, offset: int | None = None, quant_level: int | None = None ) -> Tensor:
		if offset is None:
			# prom
			if self.capabilities is None:
				offset = 0
			# resp
			elif "len" in self.capabilities:
				offset = 1
			elif "nar" not in self.capabilities:
				offset = 0
			elif quant_level > 0:
				offset = 1

		if quant_level is None:
			quant_level = 0 if xi.dim() == 1 else xi.shape[-1] - 1
		
		if self.sums and quant_level > 0:
			x = sum( [ self.embeddings[k + offset]( xi[:, k] ) for k in range( quant_level ) ] )
		else:
			k = quant_level
			x = self.embeddings[k + offset]( xi if xi.dim() == 1 else xi[:, k] )

		return x

	def forward(self, xi: Tensor, offset: int | None = None, quant_level: int | None = None ) -> Tensor:
		x = self.internal_forward( xi, offset = offset, quant_level = quant_level ) if self.external_mode != "exclusive" or xi.shape[0] == 0 else None

		if self.external_mode and xi.shape[0] > 0:
			external_embeddings = self.external_embeddings( xi, quant_level = quant_level )
			if self.external_mode == "exclusive":
				return external_embeddings
			x += external_embeddings

		return x

# per-level classification
class AudioClassifier(nn.Module):
	def __init__(
		self,
		l_tokens: list[int], # list of number of tokens (needed because AR resps includes stop token)
		token_dim: int, # dimensionality of the embedding
	):
		super().__init__()
		self.proj = nn.ModuleList([nn.Linear(token_dim, n_tokens) for n_tokens in l_tokens])

	def forward(self, xi: Tensor, levels: list[int] ) -> Tensor:
		dtype = xi.dtype
		device = xi.device

		xi = [ self.proj[l]( x ) for x, l in zip(xi, levels) ]
		# pad if needed
		# to-do: validate that this causes ZERO issues
		max_size = max([ x.shape[-1] for x in xi ])
		xi = [
			#x if l == 0 else
			x if x.shape[-1] == max_size else
			torch.cat( [ x, torch.tensor( [[ -float("inf") ] for _ in range(x.shape[0])], device=device, dtype=dtype) ] * (max_size - x.shape[-1]), dim=-1 )
			for x, l in zip(xi, levels)
		]
		return torch.stack( xi )

class Metrics(nn.Module):
	def __init__(
		self,
		l_tokens: int | list[int],
		top_k = 10,
		average="micro",
		multidim_average="global",
		ignore_index = -100
	):
		super().__init__()
		self.accuracy = nn.ModuleList([ MulticlassAccuracy(
			n_tokens,
			top_k=top_k,
			average=average,
			multidim_average=multidim_average,
			ignore_index=ignore_index,
		) for n_tokens in l_tokens ])
		self.precision = nn.ModuleList([ MulticlassPrecision(
			n_tokens,
			top_k=top_k,
			average=average,
			multidim_average=multidim_average,
			ignore_index=ignore_index,
		) for n_tokens in l_tokens ])

	def calc_accuracy( self, inputs, targets, quant_levels ):
		return sum( [ self.accuracy[l]( input[:, :self.accuracy[l].num_classes], target ) for target, input, l in zip( targets, inputs, quant_levels ) ] ) / len( inputs )
	
	def calc_precision( self, inputs, targets, quant_levels ):
		return sum( [ self.precision[l]( input[:, :self.precision[l].num_classes], target ) for target, input, l in zip( targets, inputs, quant_levels ) ] ) / len( inputs )

	def __call__(self, *args, **kwargs):
		return dict(
			acc=self.calc_accuracy(*args, **kwargs),
		)

class Base(nn.Module):
	def loss_factor(self, k):
		if self.config is None:
			return 1.0
		return self.config.loss_factors[k] if k in self.config.loss_factors else 1.0

	def _prune(self, l: Tensor, stop = None):
		if stop is None:
			stop = self.stop_token

		indices = (l == stop).nonzero()

		if len(indices) == 0:
			return l

		return l[: indices.min().item()]

	# these probably need to live in an interleaved model, as pattern-ing is targeted for a sole AR model
	"""
	def codes_to_pattern(self, codes):
		# expand if not batched
		if codes.dim() == 2:
			codes = codes.unsqueeze(0)
		# [batch, timestep, rvq level] (B, T, K) =>  [batch, rvq level, timestep] (B, K, T)
		codes = codes.permute(0, 2, 1)
		
		B, K, T = codes.shape
		
		# map codes [B, K, T] into pattern sequence [B, K, S] using special_token_id for masked tokens
		pattern = self.pattern_provider.get_pattern(T)
		sequence_codes, sequence_indexes, sequence_mask = pattern.build_pattern_sequence(
			codes.contiguous(), self.stop_token, keep_only_valid_steps=False,
		)

		# (B, K, T) => (B, T, K)
		return sequence_codes.permute(0, 2, 1)

	def logits_from_pattern(self, logits, pattern):
		logits = logits.permute(0, 3, 1, 2)  # [B, card, K, S]

		logits, logits_indexes, logits_mask = pattern.revert_pattern_logits(
			logits, float('nan'), keep_only_valid_steps=False
		)
		logits = logits.permute(0, 2, 3, 1)  # [B, K, T, card]
		logits_mask = logits_mask[None, :, :].expand(B, -1, -1)  # [K, T] -> [B, K, T]

		return logits, logits_mask
	"""

	def __init__(
		self,
		
		n_text_tokens: int = 256,
		n_audio_tokens: int = 1024,

		d_model: int = 512,
		n_heads: int = 8,
		n_layers: int = 12,
		p_dropout: float = 0.1,

		n_experts: int = 1,

		l_padding: int = 0,

		training = True, 
		config = None, 
	):
		super().__init__()
		self.training = training
		self.config = config

		self.n_text_tokens = n_text_tokens
		self.n_audio_tokens = n_audio_tokens

		self.d_model = d_model
		self.n_heads = n_heads
		self.n_layers = n_layers
		self.n_experts = n_experts
		
		self.l_padding = l_padding

		self.ignore_index = -100

		self.n_resp_levels = self.config.resp_levels if self.config else n_resp_levels
		self.n_max_levels = self.config.max_levels if self.config else n_resp_levels
		self.capabilities = self.config.capabilities if self.config else ["ar", "nar"]
		self.gradient_checkpointing = self.config.gradient_checkpointing if self.config is not None else True

		self.stop_token = self.n_audio_tokens # id 1024
		self.causal = "ar" in self.capabilities or "len" in self.capabilities
		self.version = self.config.version if self.config is not None else 5
		self.causal_size = self.config.experimental.causal_size if self.config is not None else (1 if "ar" in self.capabilities else 0)

		self.arch_type = self.config.arch_type if self.config is not None else "llama"

		# check if requested arch is unavailable
		if self.arch_type in ERROR_ARCHES:
			raise ERROR_ARCHES[self.arch_type]
		
		attention_backend = self.config.attention if self.config is not None else "auto"
		audio_embedding_sums = self.config.experimental.audio_embedding_sums if self.config is not None else False
		split_classifiers = self.config.experimental.split_classifiers if self.config is not None else False
		tie_classifier_to_embedding = self.config.experimental.tie_classifier_to_embedding if self.config is not None else False
		audio_embedding_mode = self.config.experimental.audio_embedding_mode if self.config is not None else ""
		unified_position_ids = self.config.experimental.unified_position_ids if self.config is not None else True
		interleave = self.config.experimental.interleave if self.config is not None else False

		n_tasks = self.config.tasks if self.config is not None else 8
		n_langs = self.config.langs if self.config is not None else 2
		n_tones = self.config.tones if self.config is not None else 1

		# pure AR
		if "nar" not in self.capabilities:
			n_resp_tokens = n_audio_tokens + 1
			l_tokens = [n_resp_tokens] * self.n_resp_levels
		# NAR-len model
		elif "len" not in self.capabilities:
			# +1 to include the stop token
			n_resp_tokens = n_audio_tokens + ( 1 if self.causal_size > 0 else 0 )
			l_tokens = [n_resp_tokens] + [n_resp_tokens - 1] * (self.n_resp_levels - 1)
		# AR+NAR model
		else:
			n_resp_tokens = n_audio_tokens
			l_tokens = [n_resp_tokens] * (self.n_resp_levels + (1 if split_classifiers else 0))

		# there seems to be a problem with the NAR-only model with non-unified position IDs.............
		"""
		if "len" in self.capabilities and not unified_position_ids:
			raise Exception("ERROR: model instability for NAR-only model when not using unified position IDs.")
		"""

		self.unified_position_ids = unified_position_ids
		self.interleave = interleave

		self.text_emb = Embedding(n_text_tokens, d_model)
		self.langs_emb = None
		self.tones_emb = None
		self.tasks_emb = None
		self.rvq_l_emb = None
		self.len_emb = None
		
		# it would be nicer for these to be a token or live inside an embedding
		self.sep = nn.Parameter(torch.randn(d_model))
		self.dropout_token = nn.Parameter(torch.zeros(d_model)) # zeros sounds nicer than randn for a special value

		if self.version == 1: # legacy
			n_audio_tokens += (n_tasks - 1) # old models have the task tokens in the prom
			self.proms_emb = MultiEmbedding(self.n_resp_levels, n_audio_tokens, d_model)
			self.resps_emb = MultiEmbedding(self.n_resp_levels, n_resp_tokens, d_model, monolithic=self.monolithic)
		elif self.version < 5:
			# [1024] * 8
			self.proms_emb = AudioEmbedding_Old(
				[n_audio_tokens] * self.n_resp_levels, d_model,
				levels=self.n_resp_levels if self.version > 3 else None,
			)
			# [1024 + STOP] + [1024] * 8
			self.resps_emb = AudioEmbedding_Old(
				l_tokens, d_model,
				levels=self.n_resp_levels if self.version > 3 else None,
			)
		else:
			self.proms_emb = AudioEmbedding(
				[n_audio_tokens] * self.n_resp_levels, d_model,
				sums=audio_embedding_sums,
				external_mode=audio_embedding_mode,
				capabilities=None,
			)
			self.resps_emb = AudioEmbedding(
				l_tokens, d_model,
				sums=audio_embedding_sums,
				external_mode=audio_embedding_mode,
				capabilities=self.capabilities,
			)

		# useless since I actually removed using these with the input processing overhaul...
		if self.version >= 3:
			self.langs_emb = Embedding(n_langs, d_model) if n_langs > 0 else None
			self.tasks_emb = Embedding(n_tasks, d_model) if n_tasks > 0 else None
		# never actually got added... I kept forgetting to classify all my audio for speaker's tone
		if self.version >= 4:
			self.tones_emb = Embedding(n_tones, d_model) if n_tones > 0 else None

		# mamba requires this if a model does both AR and NAR tasks
		# this *might* help for AR and NAR tasks since we explicitly specify the current RVQ level for a sequence, rather than having it "encoded" in the embeddings
		# this ***might*** let me also unify the proms_emb and resps_embedding
		if self.version >= 5:
			self.rvq_l_emb = Embedding(self.n_resp_levels + (1 if "len" in self.capabilities else 0), d_model)
		
			# experimental NAR-only mode
			self.len_emb = Embedding(11, d_model) if "len" in self.capabilities else None

		# there seems to have been a regression where anything touching the wrapped LlamaAttention class breaks

		if attention_backend == "auto":
			if "flash_attention_2" in AVAILABLE_ATTENTIONS:
				attention_backend = "flash_attention_2"
			elif "flash" in AVAILABLE_ATTENTIONS:
				attention_backend = "flash"
			elif "mem_efficient" in AVAILABLE_ATTENTIONS:
				attention_backend = "mem_efficient"
			elif "math" in AVAILABLE_ATTENTIONS:
				attention_backend = "math"
			else:
				attention_backend = "sdpa"

		if attention_backend == "xformers":
			attention_backend = "mem_efficient"
		
		hf_attention = attention_backend

		if attention_backend in ["xformers", "mem_efficient", "math", "flash", "cudnn"]:
			hf_attention = None
			if attention_backend not in AVAILABLE_ATTENTIONS:
				raise ValueError(f"Requesting attention `{attention_backend}` but is not available. Currently available: {AVAILABLE_ATTENTIONS}")

		if self.arch_type == "transformer":
			self.sin_emb = SinusoidalEmbedding(d_model)
			self.blocks = nn.ModuleList([TransformerBlock(
				d_model=d_model,
				n_heads=n_heads,
				p_dropout=p_dropout if training else 0.0,
				causal=self.causal,
				norm_type="ln", # adaln
				n_levels=self.n_resp_levels,
			) for _ in range(n_layers) ])
		elif self.arch_type in ["mistral", "mixtral"]:
			if n_experts <= 1:
				self.model = MistralModel(MistralConfig(
					vocab_size=n_resp_tokens,
					hidden_size=d_model,
					max_position_embeddings=75 * 60 * 5, # max-length of 60 seconds
					intermediate_size=d_model*4,
					num_hidden_layers=n_layers,
					num_attention_heads=n_heads,
					attention_dropout=p_dropout if training else 0.0,
					num_key_value_heads=self.config.experimental.kv_heads if self.config is not None and self.config.experimental.kv_heads > 0 else n_heads,
					hidden_act="gelu",
					is_encoder_decoder=False,
					is_decoder=True,
					attn_implementation=hf_attention,
					#gradient_checkpointing=self.gradient_checkpointing,
				))
			else:
				self.model = MixtralModel(MixtralConfig(
					vocab_size =n_resp_tokens,
					hidden_size=d_model,
					max_position_embeddings=75 * 60 * 5, # max-length of 60 seconds
					intermediate_size=d_model*4,
					num_hidden_layers=n_layers,
					num_attention_heads=n_heads,
					attention_dropout=p_dropout if training else 0.0,
					num_key_value_heads=self.config.experimental.kv_heads if self.config is not None and self.config.experimental.kv_heads > 0 else n_heads,
					sliding_window=75 * 12, # 12 second context window
					output_router_logits=training,
					hidden_act="gelu",
					is_encoder_decoder=False,
					is_decoder=True,
					num_local_experts=n_experts,
					num_experts_per_tok=min(2, n_experts),
					attn_implementation=hf_attention,
					#gradient_checkpointing=self.gradient_checkpointing,
				))
				if attention_backend in ["mem_efficient", "math", "flash", "cudnn", "auto"]:
					self.model = ml.replace_attention( self.model, klass=MixtralAttention_Adapted, target=MixtralAttention, mode=attention_backend )

			if self.gradient_checkpointing and not self.model.gradient_checkpointing:
				self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(
					use_reentrant=False
				))
		elif self.arch_type == "llama":
			if n_experts <= 1:
				self.model = LlamaModel(LlamaConfig(
					vocab_size=n_resp_tokens,
					hidden_size=d_model,
					max_position_embeddings=75 * 60 * 5, # max-length of 60 seconds
					intermediate_size=d_model*4,
					num_hidden_layers=n_layers,
					num_attention_heads=n_heads,
					attention_dropout=p_dropout if training else 0.0,
					num_key_value_heads=n_heads,
					sliding_window=75 * 12, # 12 second context window
					hidden_act="gelu",
					is_encoder_decoder=False,
					is_decoder=True,
					attn_implementation=hf_attention,
					#gradient_checkpointing=self.gradient_checkpointing,
				))
				if attention_backend in ["mem_efficient", "math", "flash", "cudnn", "auto"]:
					self.model = ml.replace_attention( self.model, klass=LlamaAttention_Adapted, target=LlamaAttention, mode=attention_backend )
			else:
				self.model = MixtralModel(MixtralConfig(
					vocab_size =n_resp_tokens,
					hidden_size=d_model,
					max_position_embeddings=75 * 60 * 5, # max-length of 60 seconds
					intermediate_size=d_model*4,
					num_hidden_layers=n_layers,
					num_attention_heads=n_heads,
					attention_dropout=p_dropout if training else 0.0,
					num_key_value_heads=n_heads,
					sliding_window=75 * 12, # 12 second context window
					output_router_logits=training,
					hidden_act="gelu",
					is_encoder_decoder=False,
					is_decoder=True,
					num_local_experts=n_experts,
					num_experts_per_tok=min(2, n_experts),
					attn_implementation=hf_attention,
					#gradient_checkpointing=self.gradient_checkpointing,
				))
				if attention_backend in ["mem_efficient", "math", "flash", "cudnn", "auto"]:
					self.model = ml.replace_attention( self.model, klass=MixtralAttention_Adapted, target=MixtralAttention, mode=attention_backend )

			if self.gradient_checkpointing and not self.model.gradient_checkpointing:
				self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(
					use_reentrant=False
				))
		elif self.arch_type == "retnet":
			kwargs = dict(
				vocab_size=n_resp_tokens,
				decoder_embed_dim=d_model,
				decoder_value_embed_dim =d_model * 2,
				decoder_retention_heads=n_heads,
				decoder_ffn_embed_dim=d_model * 4,
				decoder_layers=n_layers,
				dropout=p_dropout if training else 0.0,
				checkpoint_activations=self.gradient_checkpointing,
				activation_fn="gelu",
				use_layernorm=self.version < 3,
				use_biases=self.version < 3,
				use_glu=self.version >= 3,

				chunkwise_recurrent=self.causal and self.causal_size > 0,
				recurrent_chunkwise_size=self.causal_size if self.causal else 0,
				no_output_layer=True,
				decoder_normalize_before=True,

				rotary_embedding_base=10000
			)

			if n_experts > 1:
				kwargs.update(dict(
					use_xmoe=True,
					moe_freq=1,
					moe_expert_count=n_experts,
					moe_gating_use_fp32=False,
				))

			self.model = RetNetDecoder(RetNetConfig(**kwargs))
		elif self.arch_type == "retnet-hf":
			kwargs = dict(
				vocab_size=n_resp_tokens,
				decoder_embed_dim=d_model,
				decoder_value_embed_dim =d_model * 2,
				decoder_retention_heads=n_heads,
				decoder_ffn_embed_dim=d_model * 4,
				decoder_layers=n_layers,
				dropout=p_dropout if training else 0.0,
				checkpoint_activations=self.gradient_checkpointing,
				activation_fn="gelu",
				use_glu=False, # self.version >= 3,

				recurrent_chunk_size=self.causal_size if self.causal else 0,
				decoder_normalize_before=True,

				deepnorm=False,
				subln=True,
			)

			self.model = RetNetDecoder_HF(RetNetConfig_HF(**kwargs))

			if self.gradient_checkpointing and not self.model.gradient_checkpointing:
				self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(
					use_reentrant=False
				))
		elif self.arch_type == "bitnet":
			self.model = BitNetTransformer(
				num_tokens=n_resp_tokens,
				dim=d_model,
				depth=n_layers,
				heads=n_heads,
				ff_mult=4,
				gradient_checkpointing=self.gradient_checkpointing,
			)
		elif self.arch_type in ["mamba","mamba2"]:
			self.model = MambaMixelModel(
				vocab_size=n_resp_tokens,
				d_model=d_model,
				n_layer=n_layers,
				d_intermediate=d_model*4,
				ssm_cfg={"layer": "Mamba2", "use_mem_eff_path": False} if self.arch_type == "mamba2" else {},
				rms_norm=True,
				fused_add_norm=True,
				residual_in_fp32=False,
				#attn_layer_idx=attn_layer_idx,
				#attn_cfg=attn_cfg,
				#initializer_cfg=initializer_cfg,
			)
			self.model.gradient_checkpointing = self.gradient_checkpointing
		elif self.arch_type in ["mamba2-hf"]:
			self.model = Mamba2Model_HF(Mamba2Config_HF(
				vocab_size=n_resp_tokens,
				hidden_size=d_model,
				max_position_embeddings=75 * 60 * 5, # max-length of 60 seconds
				expand=4,
				num_hidden_layers=n_layers,
				is_encoder_decoder=False,
				is_decoder=True,
				use_triton_kernels=False, # the entire reason is to NOT use triton (because V100s hate it)
				residual_in_fp32=False, # breaks for AMP inference
			))
			if self.gradient_checkpointing and not self.model.gradient_checkpointing:
				self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(
					use_reentrant=False
				))
		elif self.arch_type == "mmfreelm":
			self.model = HGRNBitModel(HGRNBitConfig(
				vocab_size=n_resp_tokens,
				hidden_size=d_model,
				max_position_embeddings=75 * 60 * 5, # max-length of 60 seconds
				intermediate_size=d_model*4,
				num_hidden_layers=n_layers,
				num_heads=n_heads,
				#hidden_act="gelu",
				#is_encoder_decoder=False,
				#is_decoder=True,
				attn_mode=hf_attention,
				#gradient_checkpointing=self.gradient_checkpointing,
			))

			if self.gradient_checkpointing and not self.model.gradient_checkpointing:
				self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(
					use_reentrant=False
				))
		else:
			raise RuntimeError(f'Unknown arch specified: {self.arch_type}')

		if hasattr( self.model, "embeddings" ):
			del self.model.embeddings


		if not split_classifiers:
			self.classifier = nn.Linear(d_model, n_resp_tokens)
			self.classifiers = None
			
			self.accuracy_metric = MulticlassAccuracy(
				n_resp_tokens,
				top_k=10,
				average="micro",
				multidim_average="global",
				ignore_index=self.ignore_index,
			)

			self.precision_metric = MulticlassPrecision(
				n_resp_tokens,
				top_k=10,
				average="micro",
				multidim_average="global",
				ignore_index=self.ignore_index,
			)

			self.metrics = None
		else:
			self.classifier = None
			self.classifiers = AudioClassifier( l_tokens, d_model )
			self.accuracy_metric = None
			self.precision_metric = None
			self.metrics = Metrics( l_tokens )

			"""
			if tie_classifier_to_embedding:
				for i, proj in enumerate( self.classifiers.proj ):
					self.classifiers.proj[i].weight = self.resps_emb.embeddings[i].weight
			"""


	def _forward(
		self,
		inputs,
		mask = None,
		position_ids = None,
		state = None,
	):
		x = inputs
		m = mask.squeeze(-1).int()
		aux_loss = None

		# HF transformer derived model
		if self.arch_type in ["llama", "mistral", "mixtral"]:
			kwargs = dict(
				attention_mask=m,
				inputs_embeds=x,
				past_key_values=state,
				position_ids=position_ids,
				use_cache=True,
			#	return_dict=True,
			)
			if self.n_experts > 1 and self.training:
				kwargs["output_router_logits"] = True

			t = self.model(**kwargs)

			x = t[0]
			
			if state is not None:
				state = t[1]
			
			if self.n_experts > 1 and self.training:
				router_logits = t[-1]
				aux_loss = self.model.config.router_aux_loss_coef * load_balancing_loss_func( router_logits, self.model.config.num_local_experts, self.model.config.num_experts_per_tok )
		elif self.arch_type == "transformer":
			# ensures we specify a quant_level for the transformer implementation's AdaLN
			l = torch.zeros((batch_size,), dtype=torch.int32) if quant_levels is None else quant_levels
			l = l.to(device)
			# inject position information
			x = self.sin_emb.add_pe(x)
			# pass our inputs through the transformer
			for block in self.blocks:
				x = block(x, m, l)
		elif self.arch_type == "retnet":
			# pass our inputs through the RetNet
			x, _ = self.model(x, incremental_state=state, token_embeddings=x, features_only=True)
			if _ is not None and "l_aux" in _ and self.n_experts > 1:
				aux_loss = torch.sum(torch.stack([ t for t in _["l_aux"] if t is not None])) * 0.001
		elif self.arch_type == "retnet-hf":
			first = state is None or len(state) == 0

			kwargs = dict(
				attention_mask=m,
				inputs_embeds=x if first else x[:, -1, :].unsqueeze(1),
				past_key_values=None if first else state,
				use_cache=True,
				forward_impl='parallel' if first else 'recurrent',
				return_dict=True,
			)

			out = self.model(**kwargs)
			x = out.last_hidden_state
			if state is not None:
				state = out.past_key_values
		elif self.arch_type in ["mamba","mamba2"]:
			x = self.model( hidden_states=x )
		elif self.arch_type == "mamba2-hf":
			first = state is None or len(state) == 0

			kwargs = dict(
				inputs_embeds=x,
				cache_params=state,
				return_dict=True,
			)

			out = self.model(**kwargs)
			x = out.last_hidden_state
			if state is not None:
				state = out.cache_params 
		elif self.arch_type == "bitnet":
			x = self.model(x)
		elif self.arch_type == "mmfreelm":
			x = self.model(
				attention_mask=m,
				inputs_embeds=x,
			)

			x = x[0]

		# output projection layer with masking
		if self.classifier is not None:
			x = self.classifier(x) * mask

		return x, state, aux_loss

	# takes a bunch of separate lists and parses them into an ordered array of tuples to guide input sequence creation
	def inputs(
		self,
		text_list: list[Tensor],
		proms_list: list[Tensor],
		resps_list: list[Tensor],

		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
		len_list: list[Tensor] | None = None,
		task_list: list[str] | None = None,

		quant_levels: int | list[int] | Tensor | None = None
	):
		device = text_list[0].device
		batch_size = len(text_list)

		inputs = [ [] for _ in range(batch_size) ]
		for i in range(batch_size):
			quant_level = quant_levels[i] if quant_levels is not None else 0
			task_type = task_list[i] if task_list is not None else "tts"

			# insert task type as a string
			inputs[i].append( ( "task", task_type ) )

			# to-do: maybe not split the below blocks up
			# might be beneficial in the event I need to use a difference sequence, such as STT tasks

			# Base-line TTS task
			# Sequence: <text><sep><rvq lvl><sep><prom><sep><resp>
			# prom /may/ include <task> tokens inside to help guide things, per SpeechX
			if f'<{task_type}>' in get_task_symmap():
				# insert the text prompt
				if text_list is not None and text_list[i] is not None:
					inputs[i].append( ( "text", text_list[i] ) )
				# insert lang token if we're trained for it
				if "lang" in self.capabilities and lang_list is not None and lang_list[i] is not None:
					inputs[i].append( ( "lang", lang_list[i] ) )
				# insert RVQ level guidance token if the model is versioned for it
				if self.rvq_l_emb is not None and not self.interleave:
					inputs[i].append( ( "quant_level", torch.tensor([ quant_level ], device=device, dtype=torch.int16) ) )
				# insert input audio prompt
				if proms_list is not None and proms_list[i] is not None:
					inputs[i].append( ( "prom", proms_list[i] ) )
				# insert tone token if we're trained for it
				if "tone" in self.capabilities and tone_list is not None and tone_list[i] is not None:
					inputs[i].append( ( "tone", tone_list[i] ) )
				# insert the current output response
				if resps_list is not None and resps_list[i] is not None:
					inputs[i].append( ( "resp", resps_list[i] ) )
		
			# Audio length prediction task
			# Sequence: <text><sep><rvq lvl><prom><sep><len>
			elif task_type == "len":
				# throw an error so we don't silently train without this
				if self.len_emb is None:
					raise Exception(f"Requesting task `{task_type}` but corresponding embedding is not defined.")

				# insert the text prompt
				if text_list is not None and text_list[i] is not None:
					inputs[i].append( ( "text", text_list[i] ) )
				# insert lang token if we're trained for it
				if "lang" in self.capabilities and lang_list is not None and lang_list[i] is not None:
					inputs[i].append( ( "lang", lang_list[i] ) )
				# technically will always be level 0 but for the sake of keeing the input formatting coherent...
				if self.rvq_l_emb is not None:
					# override to 0 (I don't know if this change propagates, I'm not familiar with when python passes by (copied) value or reference)
					quant_levels[i] = 0
					inputs[i].append( ( "quant_level", torch.tensor([ self.n_resp_levels ], device=device, dtype=torch.int16) ) )
				# insert input audio prompt
				if proms_list is not None and proms_list[i] is not None:
					inputs[i].append( ( "prom", proms_list[i] ) )
				# insert tone token if we're trained for it
				if "tone" in self.capabilities and tone_list is not None and tone_list[i] is not None:
					inputs[i].append( ( "tone", tone_list[i] ) )

				# insert output length tokens (if it exists)
				if len_list is not None and len_list[i] is not None:
					inputs[i].append( ( "len", len_list[i] ) )
				# "encode" length to tokens for 0-9 + stop
				elif resps_list is not None and resps_list[i] is not None:
					# yes this could be encoded better
					inputs[i].append( ( "len", torch.tensor([ 0 ] + [ int(i) for i in str( resps_list[i].shape[0]) ] + [ 10 ], device=device, dtype=torch.int16) ) )
			else:
				raise Exception(f'Unrecognized task: {task_type}')

		return inputs

	def inputs_to_embeddings(
		self,
		inputs: list,
		quant_levels: int | list[int] | Tensor | None = None
	):
		# handles tasks where the prompt has task tokens injected in the middle
		def prompt_input_to_embedding( input, quant_level ):
			if isinstance(input, str):
				return self.tasks_emb( torch.tensor( [ get_task_symmap()[f'<{input}>'] ], device=device, dtype=torch.int16) )

			# get RVQ level 0, or up to targetted RVQ level inference
			if self.version <= 4:
				return self.proms_emb(
					input if quant_level == 0 else input[:, :quant_level]
				)
				
			return self.proms_emb(
				input if input.dim() == 1 else input[:, : 1 if quant_level == 0 else quant_level],
				quant_level = 0 if quant_level == 0 else quant_level - 1, # input is one below the target quant level
				offset = 0,
			)

		# yuck
		token_dropout_rate = self.config.experimental.token_dropout_rate if self.config else 0.0
		token_dropout_rvq_levels = self.config.experimental.token_dropout_rvq_levels if self.config else None
		
		if self.dropout_token is None or not self.training:
			token_dropout_rate = 0.0

		if not token_dropout_rvq_levels:
			token_dropout_rvq_levels = [1, self.resp_levels]

		x_list = []
		for batch_index, batch_input in enumerate(inputs):
			batch = []
			quant_level = quant_levels[batch_index] if quant_levels is not None else 0
			
			task_type = "tts"
			input_prom = None
			for name, input in batch_input:
				# technically can provide a map for input_name => embedding, but some embedding requires additional processing
				embedding = None

				# is already an embedding		
				if name == "task":
					# noop
					# *maybe* inject a token for specifying task type
					task_type = input
					continue
				elif name == "text":
					embedding = self.text_emb( input )

					device = embedding.device
				elif name == "quant_level" and self.rvq_l_emb is not None:
					embedding = self.rvq_l_emb( input )
				elif name == "lang" and self.langs_emb is not None:
					embedding = self.langs_emb( input )
				elif name == "prom":
					proms = [ input ] if isinstance(input, torch.Tensor) else input
					input_prom = torch.cat([ prom for prom in proms if isinstance(input, torch.Tensor) ])

					embedding = torch.cat( [ prompt_input_to_embedding( input, quant_level ) for input in proms if input is not None ] )
				elif name == "tone" and self.tones_emb is not None:
					embedding = self.tones_emb( input )
				elif name == "resp":
					if self.interleave:
						embeddings = [ self.resps_emb(
							input[:, :l+1],
							offset = 0,
							quant_level = l
						) for l in range( input.shape[-1] ) ]

						embedding = _interleave_sequence_reshape( embeddings )
					elif "len" in self.capabilities and quant_level == 0:
						if input_prom is not None:
							# fill with the prom as the initial condition
							repeat = (input.shape[0] // input_prom.shape[0]) + 1
							repeated = input_prom[:, :1].repeat((repeat, 1))[:input.shape[0], :1]

							embedding = self.resps_emb(
								repeated,
								offset = 0,
								quant_level = 0,
							)
						else:
							# fill with "stop" token from the len layer for the NAR-only model
							filler_token = 12
							embedding = self.resps_emb(
								# self.dropout_token.repeat((input.shape[0], 1)),
								torch.full_like(input if input.dim() == 1 else input[..., 0], filler_token),
								offset = 0,
								quant_level = 0,
							)

					else:
						# get RVQ level 0, or up to targetted RVQ level inference
						if self.version <= 4:
							embedding = self.resps_emb(
								input if quant_level == 0 else input[:, :quant_level],
								quant_level
							)
						else:
							offset = 0
							if "len" in self.capabilities:
								offset = 1
							elif "nar" not in self.capabilities:
								offset = 0
							elif quant_level > 0:
								offset = 1

							embedding = self.resps_emb(
								input if input.dim() == 1 or quant_level == 0 else input[:, :quant_level],
								offset = offset,
								quant_level = 0 if quant_level == 0 else quant_level - 1, # input is one below the target quant level
							)

						# apply token dropout
						if token_dropout_rate > 0.0 and (token_dropout_rvq_levels[0] <= quant_level and quant_level <= token_dropout_rvq_levels[1]):
							steps = embedding.shape[0] - (1 if quant_level == 0 else 0) # do not mess with stop token
							for i in range( steps ):
								if random.random() > token_dropout_rate:
									continue
								
								embedding[i] = self.dropout_token

				elif name == "len" and self.len_emb is not None:
					embedding = self.len_emb( input )
				else:
					# should probably raise an exception so things aren't processed silently
					continue
				batch.append(embedding)

			x_list.append( _join( batch, self.sep ) )

		return x_list

	# creates position ids from a given input list
	# if not unified_position_ids, then each input segment will have its own sequence
	def inputs_to_position_ids(
		self,
		inputs: list,
		mask: Tensor,
	):
		device = mask.device

		# shamelessly grabbed from modeling_llama.py
		ids = mask.long().cumsum(-1) - 1
		ids.masked_fill_( mask == 0, 1 )

		# there's a better way
		if not self.unified_position_ids:
			x_list = []

			def get_input_token_length( name, input ):
				# task token
				if isinstance(input, str):
					return 1

				# list of tokens
				if not isinstance(input, torch.Tensor):
					return sum( [ i.shape[0] for i in input if isinstance(i, torch.tensor) ] ) + 1

				# interleaved model
				if self.interleave and name == "resp":
					return input.shape[0] * input.shape[1]

				# ending input will not have a separator later
				return input.shape[0] + (0 if name in ["resp", "len"] else 1)

			for batch_index, batch_input in enumerate(inputs):
				batch = torch.cat( [
					torch.tensor([*range(get_input_token_length(name, input))], device=device, dtype=torch.int32)
					for name, input in batch_input if name != "task"
				] )

				delta = ids[batch_index].shape[0] - batch.shape[0]
				if delta > 0:
					batch = torch.cat( [ batch, torch.tensor([1] * delta, device=device, dtype=torch.int32) ] )

				x_list.append( batch )

			ids = torch.stack( x_list )

		return ids.to(device=device, dtype=torch.int32)

	def calc_loss(
		self,
		inputs: list,
		logits,
		
		quant_levels: int | list[int] | Tensor | None = None,
	):
		device = logits[0].device
		classifier_quant_levels = quant_levels if self.classifier is not None else [ -1 if inputs[i][0][-1] == "len" else l for i, l in enumerate( quant_levels ) ] 

		# handles tasks where the prompt has task tokens injected in the middle
		def prompt_input_to_token( input, quant_level ):
			if isinstance(input, str):
				return torch.tensor( [ get_task_symmap()[f'<{input}>'] ], device=device, dtype=torch.int16)

			# ignore prom, fill with mock tokens, because the prom embeddings don't directly map to tokens
			if self.version < 4 or (self.version >= 5 and self.config and self.config.experimental.audio_embedding_sums):
				return torch.full_like(input[..., 0], self.ignore_index)
				
			return input if input.dim() == 1 else input[:, quant_level]

		# old, "naive" way, no loss factoring
		if not self.config.loss_factors:
			target_list = []
			task_list = []

			for batch_index, batch in enumerate(inputs):
				quant_level = quant_levels[batch_index]
				target = []
				for name, input in batch:
					if name == "task":
						task_list.append( input )
					elif name == "prom":
						proms = [ input ] if isinstance(input, torch.Tensor) else input
						target.append( torch.cat( [ prompt_input_to_token( input, quant_level ) for input in proms if input is not None ] ) )
					elif name == "resp":
						if self.interleave:
							target.append( _interleave_sequence_flatten( [ input[:, l] for l in range( input.shape[-1] ) ] ) )
						else:
							target.append( input if input.dim() == 1 else input[:, quant_level] )
					elif name in ["text", "quant_level", "lang", "tone", "len"]:
						target.append( input )

				target_list.append( _join( target, torch.tensor(self.ignore_index, device=target[-1].device) ) )

			batch_size = len(target_list)
			# modify only for the AR so it can properly behave like a transformer
			for i in range(batch_size):
				quant_level = quant_levels[i]
				task_name = task_list[i]

				causal = False

				if "len" in self.capabilities:
					causal = task_name == "len"
					if quant_level >= self.n_resp_levels:
						quant_level = 0
				else:
					causal = (quant_level == 0 and "ar" in self.capabilities) or ("nar" not in self.capabilities)

				if causal:
					l = self.causal_size
					logits[i] = logits[i][..., :-l, :] # shift the target so that token n...
					target_list[i] = target_list[i][..., l:] # predicts token n + 1

			# see comments for the split-loss calc cross_entropy call
			if False:
				target = torch.cat( target_list )
				inputs = torch.cat( logits )
				self.loss = dict(
					# "nll" was in the original implementation and should actually just be called something else
					nll = F.cross_entropy( inputs, target, ignore_index=self.ignore_index )
				)
				self.stats = self.metrics( inputs, targets, classifier_quant_levels ) if self.metrics is not None else dict(
					acc = self.accuracy_metric( inputs, target ),
					# precision = self.precision_metric( inputs, target ),
				)
			else:
				self.loss = dict(
					nll = sum([ F.cross_entropy( inputs, targets, ignore_index=self.ignore_index ) for targets, inputs in zip( target_list, logits ) ]) / batch_size
				)
				self.stats = self.metrics( logits, target_list, classifier_quant_levels ) if self.metrics is not None else dict(
					acc = sum( [ self.accuracy_metric( inputs, targets ) for targets, inputs in zip( target_list, logits ) ] ) / batch_size
				)

			return

		"""
		# considerations:
		# * split losses does not maintain the entire sequence
		# * the first token is ignored for all pieces, rather than just the first text token (which is always provided)
		#	 + the other way at least should keep it intact this way
		#	 + extra logic might be required to instead offset from the end for the resp, rather than fit snuggly
		#	 + this might just be a spook since the odds the very first token of the AR mattering is slim (although I swear I hear a very brief audio pop sometimes)
		"""
		self.loss = dict()
		self.stats = dict(acc = dict())

		info = {}
		batch_size = len( inputs )

		for i, batch in enumerate( inputs ):
			quant_level = quant_levels[i]

			it = 0

			task_name = None
			for name, input in batch:
				# do not use resp
				if name == "resp":
					input = input if input.dim() == 1 else input[:, quant_level]
				# select prom level
				elif name == "prom":
					proms = [ input ] if isinstance(input, torch.Tensor) else input
					input = torch.cat( [ prompt_input_to_token( input, quant_level ) for input in proms ] )
				# meta-input, no corresponding token at the moment
				elif name == "task":
					task_name = input
					continue

				seq_len = input.shape[0]

				logit = logits[i][it:it+seq_len]
				it += seq_len + 1 # +1 to incorporate the separator
				
				causal = False
				if "len" in self.capabilities:
					causal = task_name == "len"
					if quant_level >= self.n_resp_levels:
						quant_level = 0
				else:
					causal = (quant_level == 0 and "ar" in self.capabilities) or ("nar" not in self.capabilities)
				
				# for the AR, shift sequence so that it predicts the next token
				#	 (the NAR predicts the next token in place, so it's not necessary to do any modifications for it)
				if causal and seq_len > 1:
					l = self.causal_size
					logit = logit[..., :-l, :]
					input = input[..., l:] # shift sequence to the right by one (or causal chunk size)

				if name not in info:
					info[name] = {
						"targets": [],
						"logits": [],
					}

				# modeling_llama.py has some comment about requiring .contiguous() but I feel it's a spook since that incurs a memory allocation
				info[name]["targets"].append( input.long() )
				info[name]["logits"].append( logit )

		for name, batch in info.items():
			loss_factor = self.loss_factor(name)

			if name not in ["text", "prom", "resp", "len"]:
				continue

			if loss_factor == 0.0:
				continue

			# "faster" if cross_entropy has speedups for processing an entire batch, but torch.cat allocates new tensors
			# to-do: set this to a var
			if False:
				targets = torch.cat( batch["targets"] ).long()
				inputs = torch.cat( batch["logits"] )
				self.loss[name] = F.cross_entropy( inputs, targets, ignore_index=self.ignore_index ) * loss_factor
				self.stats["acc"][name] = self.accuracy_metric( inputs, targets )
			# probably consumes less memory due to not having to allocate memory
			# this method also opens the way to scale loss per RVQ level (although it shouldn't really be needed)
			else:
				self.loss[name] = sum([ F.cross_entropy( inputs, targets, ignore_index=self.ignore_index ) * loss_factor for targets, inputs in zip( batch["targets"], batch["logits"] ) ]) / batch_size
				if self.metrics is not None:
					metrics = self.metrics( batch["logits"], batch["targets"], classifier_quant_levels )
					self.stats["acc"][name] = metrics["acc"]
				else:
					self.stats["acc"][name] = sum( [ self.accuracy_metric( inputs, targets ) for targets, inputs in zip( batch["targets"], batch["logits"] ) ] ) / batch_size

	def forward(
		self,
		inputs: list,

		quant_levels: int | list[int] | Tensor | None = None,
		state: dict | list | None = None,
	):
		x_list = self.inputs_to_embeddings( inputs, quant_levels )
		x, m = list_to_tensor(x_list)

		training = self.training
		# yes, there's a better way.
		"""
		training = False
		for batch_index, batch in enumerate(inputs):
			for name, input in batch:
				if name == "targ":
					training = True
		"""

		device = x.device
		batch_size = len(x_list)


		# pure AR
		if quant_levels is None:
			quant_levels = [ 0 for _ in range(batch_size) ]
		
		# pad our input and mask, but retain the original length by doing it after
		if self.l_padding and x.shape[1] % self.l_padding != 0:
			# pad input
			shape = list(x.shape)
			shape[1] = self.l_padding - shape[1] % self.l_padding

			padding = torch.zeros(shape, dtype=x.dtype, device=x.device)
			x = torch.cat([x, padding], dim=1)

			# pad mask
			shape[2] = 1
			padding = torch.zeros(shape, dtype=x.dtype, device=x.device)
			m = torch.cat([m, padding], dim=1)

		# needs to be done here as we still have our raw inputs
		position_ids = self.inputs_to_position_ids( inputs, mask=m.squeeze(-1).int() ) if not self.unified_position_ids else None

		x, state, aux_loss = self._forward(
			inputs=x,
			mask=m,
			state=state,
			position_ids=position_ids,
		)

		if self.classifiers is not None:
			classifier_quant_levels = quant_levels if self.classifier is not None else [ -1 if inputs[i][0][-1] == "len" else l for i, l in enumerate( quant_levels ) ] 
			x = self.classifiers(x, levels = classifier_quant_levels) * m

		# Remove padding
		logits = [ hi[:li] for hi, li in zip(x, map(len, x_list)) ]
		
		# compute loss if the target is given
		if training:
			self.calc_loss( inputs=inputs, logits=logits, quant_levels=quant_levels )

			# include any additional losses (for example: MoE router)
			if aux_loss is not None:
				self.loss["aux_loss"] = aux_loss
			
		return (logits, state) if state is not None else logits

	def sample(
		self,
		logits: list[Tensor], # logit scores
		resps_list: list[Tensor], # previous tokens
		quant_levels: int | list[int] | Tensor | None = None,
		# base sampling parameters
		temperature: float = 1.0,
		min_temperature: float = -1.0, # activates dynamic temperature sampling
		top_k: int = -100,
		top_p: float = 1.0,
		# repetition penalty parameters
		repetition_penalty: float = 1.0,
		repetition_penalty_decay: float = 0.0,
		# length penalty parameters
		length_penalty: float = 0.0,
		# beam sampling parameters
		beam_width: int = 0,
		# mirostat sampling parameters
		mirostat: list[dict] | None = None,
		# DRY sampling parameters
		dry_multiplier=0.0,
		dry_base=1.75,
		dry_allowed_length=2,
	):
		if min_temperature < 0:
			min_temperature = temperature

		# (NAR) return the entire generated response
		# Parallel decoding relies on the last N tokens in the logits, because each token predicts the next RVQ layer in the same place (forgetfully obviously)		
		if quant_levels is not None: #  and "nar" in self.capabilities: # for when I get around to coping about dropping the NAR entirely
			logits = [ logit[-l:] for logit, l in zip(logits, map(len, resps_list)) ]
		# (AR chunkwise) return the last chunkwise piece
		elif self.causal:
			logits = [ logit[-self.causal_size:] for logit in logits ]

		devices = [ logit.device for logit in logits ]
		logits = [ logit.to(device="cpu", dtype=logit.dtype if logit.dtype != torch.float16 else torch.float32) for logit in logits ]
		
		# (NAR) disable stop token
		if quant_levels is not None and "ar" in self.capabilities:
			logits = [ ban_tokens(logit, tokens=[self.stop_token]) for logit, l in zip( logits, map(len, resps_list) ) ]
		# (AR-len) disable extraneous tokens
		if quant_levels is None and "len" in self.capabilities:
			logits = [ ban_tokens(logit, tokens=[*range(11, logit.shape[-1])]) for logit, l in zip( logits, map(len, resps_list) ) ]

		# argmax instead
		if temperature <= 0.0:
			return [ logit.argmax(dim=1) for logit in logits ]

		# perform repetition penalizing	
		if "len" not in self.capabilities:
			logits = [ reptition_penalize(logit, previous=resps[:, -1].tolist(), factor=repetition_penalty, decay=repetition_penalty_decay) for logit, resps in zip( logits, resps_list ) ]

		# (AR) perform length penalizing
		if quant_levels is None and self.causal:
			logits = [ length_penalize(logit, length=l + 1, factor=length_penalty, token=self.stop_token) for logit, l in zip( logits, map(len, resps_list) ) ]

		# perform top_k/top_p filtering of our logits
		if top_k > 0 or top_p < 1.0:
			logits = [ top_k_top_p_filtering(logit, top_k=top_k, top_p=top_p) for logit in logits ]	

		# trigger dynamic temperature sampling if the minimum temperature is not the same as the sampling temperature
		#	 epsilon float comparison because I don't trust Python
		if abs(temperature - min_temperature) >= 0.001: 
			logits = [ dynamic_temperature(logit, temperature=temperature, min_temperature=min_temperature) for logit in logits ]
		else:
			logits = [ logit / temperature for logit in logits ]

		# do DRY sampling
		if dry_multiplier > 0.0:
			logits = [ dry_sampling(logit, previous=resps[:, -1].tolist(), factor=dry_multiplier, base=dry_base, allowed_length=dry_allowed_length) for logit, resps in zip( logits, resps_list ) ]

		# do mirostat sampling
		# currently incompatible with beam searching with the way the two are implemented, perhaps a night of brain bashing can make the two work
		if mirostat is not None:
			# mirostat sampling
			return [ mirostat_sample(logit, state=state) for logit, state in zip(logits, mirostat) ]

		# do beam search (naive implementation)
		# picks the top-k across all batches, and re-batches those resultant tokens
		# returns the logit scores as well to be P-concatted with the previous scores
		# to-do: not naively implement beam searching
		if beam_width > 1:
			candidates = top_k_logits_list( logits, beam_width )
			res = [ torch.tensor(token, dtype=torch.int16).unsqueeze(dim=-1) for batch, token in candidates ]
			scores = [ logits[batch].flatten()[token] for batch, token in candidates ]
			return res, scores

		# and sample
		return [ Categorical(logits=logit).sample() for logit in logits ]