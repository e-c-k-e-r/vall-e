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

from time import perf_counter
from collections import namedtuple
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
from ..utils import wrapper as ml, clamp
from ..samplers import *
from ..emb.qnt import encode_as_embedding

# yuck, kind of needed
from ..data import get_task_symmap

# these seem more elegant than a dict
Logits = namedtuple('Logits', ['logits', 'state', 'loss', 'attentions', 'hidden_states', 'exited_layer'])
Sampled = namedtuple('Sampled', ['ids', 'logits', 'scores', 'entropy'])
LossStats = namedtuple('LossStats', ['loss', 'stats'])

"""
from ..utils.pattern import DelayedPatternProvider, VALLEPattern
"""

summed_embeddings_task = [ "stt" ]
special_tasks = [ "len", "stt" ]
non_tokened_names = ["task", "dropout_mask", "classifier_level"]
task_outputs = {
	"tts": "resp",
	"stt": "text",
	"len": "len",
}

def _dropout_mask( input, p=None ):
	# cosine scheduling
	if p is None:
		t = random.random()
		p = math.cos(t * math.pi * 0.5)

	seq = [ random.random() < p for _ in range( input.shape[0] ) ]
	mask = torch.tensor( seq, dtype=torch.bool, device=input.device )
	return mask

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
	"""
	m = m.t().unsqueeze(-1)  # (t b 1)
	m = rearrange(m, pattern)
	"""
	m = m.to(x).int()
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
		l_names: list[str] = [], # names to map to indices
	):
		super().__init__()
		# array of embeddings
		#   proms are [0, resp_levels]
		#   resp are split to where [0] is for the AR, and [1:] are reserved for NAR
		self.embeddings = nn.ModuleList([nn.Embedding(n_tokens, token_dim) for n_tokens in l_tokens])
		# further experimentation is needed to see if this actually is useful
		self.sums = sums
		# 
		self.names = l_names

	def forward(self, xi: Tensor, offset: int | None = None, quant_level: int | None = None, name: str | None = None, sums = None ) -> Tensor:
		if sums is None:
			sums = self.sums
		
		if quant_level is None:
			quant_level = 0 if xi.dim() == 1 else xi.shape[-1] - 1

		# handle mapping from name
		if name in self.names:
			offset = self.names.index( name )
			offset -= quant_level # offset by quant level since it'll iterate up that many levels
		
		if self.sums and quant_level > 0:
			x = sum( [ self.embeddings[k + offset]( xi[:, k] ) for k in range( quant_level ) ] )
		else:
			k = quant_level
			x = self.embeddings[k + offset]( xi if xi.dim() == 1 else xi[:, k] )

		return x

# time-step embedding
# for the NAR-len, since it probably most likely requires encoding the timestep
class TimeEmbedding(nn.Module):
	def __init__(
		self,
		d_model
	):
		super().__init__()
		self.emb = SinusoidalEmbedding(d_model)
		self.mlp = nn.Sequential(
			nn.Linear(d_model, d_model*4),
			nn.SiLU(),
			nn.Linear(d_model*4, d_model),
		)

	def forward( self, t ):
		t = self.emb(t)
		t = self.mlp(t)

		return t

# per-level classification
# it might actually be "better" in the long run to only have one output head like a traditional LM, and just de-stitch it here instead of doing modulus math and whatever like the HF/experimental impl
class Classifiers(nn.Module):
	def __init__(
		self,
		l_tokens: list[int], # list of number of tokens (needed because AR resps includes stop token)
		token_dim: int, # dimensionality of the embedding
		l_names: list[str] | None = None, # list of names to map to each classifier
	):
		super().__init__()
		self.proj = nn.ModuleList([nn.Linear(token_dim, n_tokens) for n_tokens in l_tokens])
		self.names = l_names

	def indices(
		self,
		names
	):
		if isinstance( names[-1], int ):
			return names
		return [ self.names.index(name) for name in names ]

	def forward(self, xi: Tensor, levels: list[int] | None = None, names: list[str] | None = None ) -> Tensor:
		dtype = xi.dtype
		device = xi.device

		if levels and isinstance( levels[-1], str ):
			names = levels
			levels = []

		# map names to levels
		if names and not levels:
			levels = [ self.names.index(name) for name in names ]

		xi = [ self.proj[l]( x ) for x, l in zip(xi, levels) ]
		# pad if needed
		# to-do: validate that this causes ZERO issues
		max_size = max([ x.shape[-1] for x in xi ])
		xi = [
			#x if l == 0 else
			x if x.shape[-1] == max_size else
			torch.cat( [x, torch.full( (x.shape[0], max_size - x.shape[-1]), -float("inf"), device=device, dtype=dtype) ], dim=-1 )
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

	def calc_accuracy( self, inputs, targets, classifier_levels ):
		return sum( [ self.accuracy[l]( input[:, :self.accuracy[l].num_classes], target ) for target, input, l in zip( targets, inputs, classifier_levels ) ] ) / len( inputs )
	
	def calc_precision( self, inputs, targets, classifier_levels ):
		return sum( [ self.precision[l]( input[:, :self.precision[l].num_classes], target ) for target, input, l in zip( targets, inputs, classifier_levels ) ] ) / len( inputs )

	def __call__(self, *args, **kwargs):
		return dict(
			acc=self.calc_accuracy(*args, **kwargs),
		)

class Base(nn.Module):
	def loss_factor(self, k):
		if self.config is None:
			return 1.0
		return self.config.loss_factor(k)

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
		attention = None,
		config = None, 
	):
		super().__init__()
		self.training = training
		self.teaching = False
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
		self.causal_size = self.config.experimental.causal_size if self.config is not None else (1 if self.causal else 0)

		self.arch_type = self.config.arch_type if self.config is not None else "llama"

		# check if requested arch is unavailable
		if self.arch_type in ERROR_ARCHES:
			raise ERROR_ARCHES[self.arch_type]
		
		if not attention:
			attention = self.config.attention if self.config is not None else "auto"

		# crunge
		if self.config is not None and config.teacher:
			self.teaching = True
			self.training = False

		attention_backend = attention
		audio_embedding_sums = self.config.experimental.audio_embedding_sums if self.config is not None else False
		split_classifiers = self.config.experimental.split_classifiers if self.config is not None else False
		tie_classifier_to_embedding = self.config.experimental.tie_classifier_to_embedding if self.config is not None else False
		audio_embedding_mode = self.config.experimental.audio_embedding_mode if self.config is not None else ""
		unified_position_ids = self.config.experimental.unified_position_ids if self.config is not None else True
		interleave = self.config.experimental.interleave if self.config is not None else False
		noncausal_masks = self.config.experimental.noncausal_masks if self.config is not None else False
		teacher_alpha = self.config.experimental.teacher_alpha if self.config is not None else 0.5
		teacher_temperature = self.config.experimental.teacher_temperature if self.config is not None else 0.5
		
		masking_ratio = self.config.experimental.masking_ratio if self.config is not None else False
		ignore_inputs_for_loss = self.config.experimental.ignore_inputs_for_loss if self.config is not None else False
		
		layerskip = self.config.experimental.layerskip if self.config is not None else False
		layerskip_r = self.config.experimental.layerskip_r if self.config is not None else 2
		layerskip_p_max = self.config.experimental.layerskip_p_max if self.config is not None else 0.1
		layerskip_e_scale = self.config.experimental.layerskip_e_scale if self.config is not None else 0.1

		n_tasks = self.config.tasks if self.config is not None else 8
		n_langs = self.config.langs if self.config is not None else 2
		n_tones = self.config.tones if self.config is not None else 1

		# pure AR
		if "nar" not in self.capabilities:
			n_resp_tokens = n_audio_tokens + 1
			l_tokens = [n_resp_tokens] * self.n_resp_levels
			resp_l_names = [f'AR:{i}:{i}' for i in range( self.n_resp_levels )]
		# NAR-len model
		elif "len" in self.capabilities:
			# +1 to include the stop or mask token
			n_resp_tokens = n_audio_tokens + ( 1 if self.causal_size > 0 else 0 )
			if "ar" in self.capabilities:
				l_tokens = [n_resp_tokens] + [n_resp_tokens - 1] * (self.n_resp_levels - 1) + [n_resp_tokens]
				resp_l_names = ['AR:0:0'] + [f'NAR:{i}:{i+1}' for i in range( self.n_resp_levels - 1 )] + ['NAR:0:0']
			else:
				l_tokens = [n_resp_tokens] + [n_resp_tokens - 1] * (self.n_resp_levels - 1)
				resp_l_names = ['NAR:0:0'] + [f'NAR:{i}:{i+1}' for i in range( self.n_resp_levels - 1 )]
		# AR+NAR model
		else:
			# +1 to include the stop or mask token
			n_resp_tokens = n_audio_tokens + ( 1 if self.causal_size > 0 else 0 )
			l_tokens = [n_resp_tokens] + [n_resp_tokens - 1] * (self.n_resp_levels - 1)
			resp_l_names = ['AR:0:0'] + [f'NAR:{i}:{i+1}' for i in range( self.n_resp_levels - 1 )]

		classifier_l_tokens = l_tokens + [ n_text_tokens ]
		classifier_l_names = resp_l_names + [ "stt" ]

		if "len" in self.capabilities:
			classifier_l_tokens += [ n_text_tokens ]
			classifier_l_names += ["len"]

		self.unified_position_ids = unified_position_ids
		self.interleave = interleave
		self.layerskip = layerskip
		self.inject_timestep_embedding = False # results in bad output
		self.masking_ratio = masking_ratio
		self.ignore_inputs_for_loss = ignore_inputs_for_loss
		self.noncausal_masks = noncausal_masks
		self.teacher_alpha = teacher_alpha
		self.teacher_temperature = teacher_temperature

		# use internal attention mechanism for now because I dont have a better way to handle mixed causal/noncausal masks for other attention backends
		"""
		if noncausal_masks:
			attention_backend = "default"
		"""

		self.text_emb = Embedding(n_text_tokens, d_model)
		self.langs_emb = None
		self.tones_emb = None
		self.tasks_emb = None
		self.rvq_l_emb = None
		self.len_emb = None
		
		# it would be nicer for these to be a token or live inside an embedding
		self.sep = nn.Parameter(torch.randn(d_model))
		self.dropout_token = nn.Parameter(torch.randn(d_model))

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
			)
			self.resps_emb = AudioEmbedding(
				l_tokens, d_model,
				sums=audio_embedding_sums,
				l_names=resp_l_names,
			)

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
			# "len" RVQ level-0 gets an additional token
			self.rvq_l_emb = Embedding(self.n_resp_levels, d_model)
		
			# experimental NAR-only mode
			self.len_emb = Embedding(11, d_model)
			self.time_emb = None # TimeEmbedding(d_model) # if not masking_ratio else None

		if attention_backend == "auto":
			attention_backend = "sdpa"
			"""
			if AVAILABLE_ATTENTIONS:
				attention_backend = AVAILABLE_ATTENTIONS[0]
			else:
				attention_backend = "default"
			"""

		hf_attention = attention_backend
		HF_ATTENTIONS = ["eager", "sdpa", "flash_attention_2"]

		if attention_backend not in HF_ATTENTIONS:
			hf_attention = None
			if attention_backend not in AVAILABLE_ATTENTIONS:
				raise ValueError(f"Requesting attention `{attention_backend}` but is not available. Currently available: {AVAILABLE_ATTENTIONS}")

		# override any requested padding size
		if attention_backend == "flash_attn_v100":
			self.l_padding = 32
		elif attention_backend == "fused_attn":
			self.l_padding = 128

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
				if attention_backend not in HF_ATTENTIONS:
					self.model = ml.replace_attention( self.model, klass=MixtralAttention_Adapted, target=MixtralAttention, mode=attention_backend )

			if self.gradient_checkpointing and not self.model.gradient_checkpointing:
				self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(
					use_reentrant=False
				))
		elif self.arch_type == "llama":
			LlamaClass = LlamaModel_Adapted # if (self.layerskip or "len" in self.capabilities) else LlamaModel

			if n_experts <= 1:
				self.model = LlamaClass(LlamaConfig(
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

				# replace with desired attention
				if attention_backend not in HF_ATTENTIONS:
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
				if attention_backend not in HF_ATTENTIONS:
					self.model = ml.replace_attention( self.model, klass=MixtralAttention_Adapted, target=MixtralAttention, mode=attention_backend )

			if self.layerskip:
				self.model.layer_dropout_p = layerskip_p_max
				self.model.early_exit_scale = layerskip_e_scale
				self.model.early_exit_r = layerskip_r

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
		elif self.arch_type in ["mamba2"]:
			self.model = Mamba2Model(Mamba2Config(
				vocab_size=n_resp_tokens,
				hidden_size=d_model,
				expand=2,
				num_hidden_layers=n_layers*2,
				residual_in_fp32=True,
			))
			if self.gradient_checkpointing and not self.model.gradient_checkpointing:
				self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(
					use_reentrant=False
				))
		elif self.arch_type in ["mamba"]:
			self.model = MambaModel(MambaConfig(
				vocab_size=n_resp_tokens,
				hidden_size=d_model,
				expand=2,
				num_hidden_layers=n_layers*2,
				residual_in_fp32=True,
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
			self.classifiers = Classifiers( classifier_l_tokens, d_model, l_names=classifier_l_names )
			self.accuracy_metric = None
			self.precision_metric = None
			self.metrics = Metrics( classifier_l_tokens )

			"""
			if tie_classifier_to_embedding:
				for i, proj in enumerate( self.classifiers.proj ):
					self.classifiers.proj[i].weight = self.resps_emb.embeddings[i].weight
			"""


	def _forward(
		self,
		inputs,
		mask = None,
		is_causal = None,
		position_ids = None,
		
		state = None,
		
		layer_skip_lambda = None,

		output_attentions = False,
		output_hidden_states = False,
	):
		x = inputs
		m = mask #.squeeze(-1).int()
		
		aux_loss = None
		attentions = None
		hidden_states = None

		# HF transformer derived model
		if self.arch_type in ["llama", "mistral", "mixtral"]:
			kwargs = dict(
				inputs_embeds=x,
				attention_mask=m,
				past_key_values=state,
				position_ids=position_ids,
				use_cache=False, # not self.training,
				output_attentions=output_attentions,
				output_hidden_states=output_hidden_states,
				return_dict=True,
				is_causal=is_causal,
			)

			if self.n_experts > 1 and self.training:
				kwargs["output_router_logits"] = True

			if self.layerskip and layer_skip_lambda is not None:
				kwargs["layer_skip_lambda"] = layer_skip_lambda

			output = self.model(**kwargs)
			x = output["last_hidden_state"]
			
			# to-do: figure out why KV caching doesn't work
			#if not self.training:
			if state is not None:
				state = output["past_key_values"]

			if output_attentions:
				attentions = output["attentions"]
			
			if output_hidden_states:
				hidden_states = output["hidden_states"]
			
			if self.n_experts > 1 and self.training:
				router_logits = output["aux_loss"]
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
		elif self.arch_type in ["mamba","mamba2"]:
			kwargs = dict(
				inputs_embeds=x,
				attention_mask=m,
				#cache_params=state,
				use_cache=False, # not self.training,
				#position_ids=position_ids,
				#output_attentions=output_attentions,
				output_hidden_states=output_hidden_states,
				return_dict=True,
			)

			output = self.model(**kwargs)
			x = output["last_hidden_state"]
			
			if state is not None:
				state = output["cache_params"]

			if output_attentions:
				attentions = output["attentions"]
			
			if output_hidden_states:
				hidden_states = output["hidden_states"]

		# process it into a format that I like
		if output_hidden_states:
			# hidden_states is actually layers + 1, as hidden_states[0] == embedding...........
			hidden_states = [ state for state in hidden_states[1:] ]
			# apply normalization to these states (to-do: check if this matters)
			# but skip the last state, as it already is normalized
			hidden_states = [ x if i == self.n_layers - 1 else self.model.norm(output.hidden_states[i]) for i, state in enumerate( hidden_states ) ]

		return Logits(x, state, aux_loss, attentions, hidden_states, None)

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
		time_list: list[Tensor] | None = None,

		quant_levels: int | list[int] | Tensor | None = None
	):
		device = text_list[0].device
		batch_size = len(text_list)

		inputs = [ [] for _ in range(batch_size) ]
		for i in range(batch_size):
			quant_level = quant_levels[i] if quant_levels is not None else 0
			task_type = task_list[i] if task_list is not None else "tts"
			timestep = time_list[i] if time_list is not None else None
			classifier_level = None

			# insert task type as a string
			inputs[i].append( ( "task", task_type ) )

			# to-do: maybe not split the below blocks up
			# might be beneficial in the event I need to use a difference sequence, such as STT tasks

			# Base-line TTS task
			# Sequence: <text><sep><rvq lvl><sep><prom><sep><resp>
			# prom /may/ include <task> tokens inside to help guide things, per SpeechX
			if f'<{task_type}>' in get_task_symmap() and task_type not in special_tasks:
				# insert the text prompt
				if text_list is not None and text_list[i] is not None:
					inputs[i].append( ( "text", text_list[i] ) )
				# insert lang token if we're trained for it
				if "lang" in self.capabilities and lang_list is not None and lang_list[i] is not None:
					inputs[i].append( ( "lang", lang_list[i] ) )
				# insert RVQ level guidance token if the model is versioned for it
				if self.rvq_l_emb is not None and not self.interleave:
					inputs[i].append( ( "quant_level", torch.tensor([ quant_level ], device=device, dtype=torch.int16) ) )

					classifier_level = "AR:0:0" if quant_level == 0 else f'NAR:{quant_level-1}:{quant_level}'
				# insert input audio prompt
				if proms_list is not None and proms_list[i] is not None:
					inputs[i].append( ( "prom", proms_list[i] ) )
				# insert tone token if we're trained for it
				if "tone" in self.capabilities and tone_list is not None and tone_list[i] is not None:
					inputs[i].append( ( "tone", tone_list[i] ) )
				# insert timestep token
				if timestep is not None:
					# force set to use this classifier level
					classifier_level = "NAR:0:0"
					# store timestep information
					if self.masking_ratio in ["random", "rand"]:
						# cosine scheduled timestep => masking ratio
						p = math.cos(timestep * math.pi * 0.5)
						# I don't think is is necessary as the timestep is encoded in the sequence by the number of masked tokens, probably.
						if self.inject_timestep_embedding:
							inputs[i].append( ("timestep", torch.tensor([timestep], device=device, dtype=self.time_emb.mlp[0].weight.dtype) ) )
					else:
						# a paper said to use a fixed masking ratio of 0.8 for training
						# ...but I want to make it user adjustable
						p = self.masking_ratio

					# store dropout mask (if training, as this gets used later to mask the input embeddings if provided)
					if self.training:
						dropout_mask = _dropout_mask( resps_list[i], p )
						inputs[i].append( ("dropout_mask", dropout_mask ) )
				# insert the current output response
				if resps_list is not None and resps_list[i] is not None:
					inputs[i].append( ( "resp", resps_list[i] ) )
		
				inputs[i].append( ("classifier_level", classifier_level) )
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
					inputs[i].append( ( "quant_level", torch.tensor([ quant_level ], device=device, dtype=torch.int16) ) )
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
				
				inputs[i].append( ("classifier_level", "len") )
			# Speech-to-Text prediction task
			# Sequence: <resp><sep><rvq lvl><sep><text>
			elif task_type == "stt":
				# insert the input response
				if resps_list is not None and resps_list[i] is not None:
					inputs[i].append( ( "resp", resps_list[i] ) )
				# insert lang token if we're trained for it
				if "lang" in self.capabilities and lang_list is not None and lang_list[i] is not None:
					inputs[i].append( ( "lang", lang_list[i] ) )
				# insert RVQ level guidance token if the model is versioned for it
				if self.rvq_l_emb is not None and not self.interleave:
					inputs[i].append( ( "quant_level", torch.tensor([ quant_level ], device=device, dtype=torch.int16) ) )
				# insert the output text prompt
				if text_list is not None and text_list[i] is not None:
					inputs[i].append( ( "text", text_list[i] ) )

				inputs[i].append( ("classifier_level", "stt") )
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
			classifier_level = None
			dropout_mask = None
			timestep = None
			
			# pre-iterate
			for name, input in batch_input:
				if name == "classifier_level":
					classifier_level = input
				elif name == "dropout_mask":
					dropout_mask = input
				elif name == "timestep":
					timestep = input

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
					"""
					if proms is None:
						continue
					"""
					# to-do: probably insert separators if task requires it?
					embedding = torch.cat( [ prompt_input_to_embedding( input, quant_level ) for input in proms if input is not None ] )
				elif name == "tone" and self.tones_emb is not None:
					embedding = self.tones_emb( input )
				elif name == "resp":
					if self.interleave:
						embeddings = [ self.resps_emb(
							input[:, :l+1],
							#offset = 0,
							#quant_level = l,
							name = 'AR:0:0' if l == 0 else f'NAR:{l-1}:{l}',
						) for l in range( input.shape[-1] ) ]

						embedding = _interleave_sequence_reshape( embeddings )

					# if training NAR-len RVQ level 0
					elif dropout_mask is not None:
						embedding = self.resps_emb(
							# if masked use masked token, else original token
							torch.where( dropout_mask, self.stop_token, input if input.dim() == 1 else input[:, 0] ),
							#quant_level = 0,
							name = classifier_level,
						)
					# NAR-len
					elif classifier_level == "NAR:0:0":
						embedding = self.resps_emb(
							input if input.dim() == 1 else input[:, 0],
							#quant_level = 0,
							name = classifier_level,
						)
					# cheat-y way to handle performing STT across all levels
					elif task_type in summed_embeddings_task:
						# we do a manual sum because I trained it to use the AR embeddings + NAR embeddings for STT......
						embedding = sum([ self.resps_emb(
							input[:, :l+1],
							offset = 0 if l == 0 else 1, # or maybe set to 1
							quant_level = l,
							#name = 'AR:0:0' if l == 0 else f'NAR:{l-1}:{l}',
							sums = False
						) for l in range( input.shape[-1] - 1 ) ])
					else:
						# get RVQ level 0, or up to targetted RVQ level inference
						if self.version <= 4:
							embedding = self.resps_emb(
								input if quant_level == 0 else input[:, :quant_level],
								quant_level
							)
						else:
							"""
							offset = 0
							if "nar" not in self.capabilities:
								offset = 0
							elif quant_level > 0:
								offset = 1

							embedding = self.resps_emb(
								input if input.dim() == 1 or quant_level == 0 else input[:, :quant_level],
								offset = offset,
								quant_level = 0 if quant_level == 0 else quant_level - 1, # input is one below the target quant level
							)
							"""

							embedding = self.resps_emb(
								input if input.dim() == 1 or quant_level == 0 else input[:, :quant_level],
								#offset = 0 if classifier_level.startswith("AR:") else 1,
								name = classifier_level,
								quant_level = 0 if quant_level == 0 else quant_level - 1, # input is one below the target quant level
							)

						# apply token dropout
						if token_dropout_rate > 0.0 and (token_dropout_rvq_levels[0] <= quant_level and quant_level <= token_dropout_rvq_levels[1]):
							steps = embedding.shape[0] - (1 if quant_level == 0 else 0) # do not mess with stop token
							for i in range( steps ):
								if random.random() > token_dropout_rate:
									continue
								
								embedding[i] = self.dropout_token
				elif name == "timestep" and self.time_emb is not None:
					embedding = self.time_emb( input )
				elif name == "len" and self.len_emb is not None:
					embedding = self.len_emb( input )
				else:
					# should probably raise an exception so things aren't processed silently
					continue

				batch.append(embedding)

			x_list.append( _join( batch, self.sep ) )

		return x_list

	# get an attribute from a given input list
	def get_input(
		self,
		inputs,
		name,
		at=None,
	):
		find_all = at is None
		res = [] if at is None else None
		
		for batch_index, batch_input in enumerate(inputs):
			if not find_all and batch_index != at:
				continue

			for n, input in batch_input:
				if n != name:
					continue
				if not find_all:
					return input
				res.append( input )
		
		return res

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

			def get_input_token_length( name, input, task ):
				# task token
				if isinstance(input, str):
					return 1

				# list of tokens
				if not isinstance(input, torch.Tensor):
					return sum( [ i.shape[0] for i in input if isinstance(i, torch.Tensor) ] )

				# interleaved model
				if self.interleave and name == "resp":
					return input.shape[0] * input.shape[1]

				# ending input will not have a separator later
				return input.shape[0]

			for batch_index, batch_input in enumerate(inputs):
				# pre-iterate
				task = "tts"
				for name, input in batch_input:
					if name == "task":
						task = input

				batch = torch.cat( [
					torch.tensor([*range(get_input_token_length(name, input, task) + (1 if name != task_outputs.get(task, name) else 0))], device=device, dtype=torch.int32)
					for name, input in batch_input if name not in non_tokened_names
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
		
		quant_levels: list[int] | None = None,
		compute_hard_loss = True,
		compute_acc = True,
	):
		loss = {}
		stats = {}

		device = logits[0].device
		batch_size = len(logits)
		classifier_levels = self.get_input( inputs, "classifier_level" )

		# handles tasks where the prompt has task tokens injected in the middle
		def prompt_input_to_token( input, quant_level ):
			"""
			if isinstance(input, str):
				return torch.tensor( [ self.ignore_index ], device=device, dtype=torch.int16)
			
			return torch.tensor( [ self.ignore_index ] * input.shape[0], device=device, dtype=torch.int16)
			"""
			if isinstance(input, str):
				return torch.tensor( [ get_task_symmap()[f'<{input}>'] ], device=device, dtype=torch.int16)

			# ignore prom, fill with mock tokens, because the prom embeddings don't directly map to tokens
			if self.version < 4 or (self.version >= 5 and self.config and self.config.experimental.audio_embedding_sums):
				return torch.full_like(input[..., 0], self.ignore_index)
				
			return input if input.dim() == 1 else input[:, quant_level]
		
		for batch_index, batch in enumerate(inputs):
			quant_level = quant_levels[batch_index]
			target = []
			causal = True
			task_type = "tts"
			dropout_mask = None
			classifier_level = None
			output_len = 0

			for name, input in batch:
				if name == "task":
					task_type = input
				elif name == "dropout_mask":
					dropout_mask = input
				elif name == "classifier_level":
					classifier_level = input

			# autoregressive, causal
			if classifier_level.startswith("AR:"):
				causal = True
			# nonautoregressive, parallel
			elif classifier_level.startswith("NAR:"):
				causal = False

			it = 0
			for name, input in batch:
				token = None
				ignored = False

				# non-tokened tasks
				if name in non_tokened_names:
					continue
				# prom can either be a tensor itself or a list of tensors and strings
				if name == "prom":
					# expand to list if not a list
					proms = [ input ] if isinstance(input, torch.Tensor) else input
					# iterate over the list to inject their tokens
					token = torch.cat( [ prompt_input_to_token( input, quant_level ) for input in proms if input is not None ] )
				elif name == "resp":
					# mask found, apply it
					if dropout_mask is not None:
						# if mask use original token, else ignore
						token = torch.where( dropout_mask, input if input.dim() == 1 else input[:, 0], self.ignore_index )
					# flatten
					elif self.interleave:
						token = _interleave_sequence_flatten( [ input[:, l] for l in range( input.shape[-1] ) ] )
					# use resps as-is
					else:
						token = input if input.dim() == 1 else input[:, quant_level]
				# not a special input, inject as-is
				else:
					token = input

				if not isinstance(token, torch.Tensor):
					continue

				if token.is_floating_point():
					ignored = True

				# grab range of our logits for later
				seq_len = token.shape[0]
				start, end = it, it+seq_len
				it += seq_len + 1 # +1 to incorporate the separator

				# deduce if a name for a task is an input or output
				if name != task_outputs.get(task_type, name):
					if self.ignore_inputs_for_loss:
						ignored = True
				else:
					output_len = seq_len

				if ignored:
					# pruned
					if self.config.loss_factors:
						continue
					# fill with ignored out tensor
					token = torch.tensor( [ self.ignore_index ] * input.shape[0], device=device, dtype=torch.int16)
					
				# perform loss calculation on the individual piece
				if self.config.loss_factors:
					loss_factor = self.loss_factor(name)

					if loss_factor == 0.0:
						continue

					logit = logits[batch_index][start:end]

					if causal and seq_len > 1:
						l = self.causal_size
						logit = logit[..., :-l, :]
						token = token[..., l:] # shift sequence to the right by one (or causal chunk size)

					if compute_hard_loss:
						nll = F.cross_entropy( logit, token.long(), ignore_index=self.ignore_index ) * loss_factor
						if f'{name}.nll' not in loss:
							loss[f'{name}.nll'] = []
						loss[f'{name}.nll'].append( nll )
					
					if compute_acc:
						if self.metrics is not None:
							metrics = self.metrics.calc_accuracy( [ logit ], [ token ], self.classifiers.indices([ classifier_level ]) )
						else:
							metrics = self.accuracy_metric( logit, token )
						if f'{name}.acc' not in stats:
							stats[f'{name}.acc'] = []
						stats[f'{name}.acc'].append( metrics )
				# add to list
				else:
					target.append( token )
			
			# perofrm loss calculation on the entire sequence
			if not self.config.loss_factors:
				target = _join( target, torch.tensor(self.ignore_index, device=target[-1].device) )
				logit = logits[batch_index]

				# shift if causal
				if causal:
					l = self.causal_size
					logit = logit[..., :-l, :] # shift the target so that token n...
					target = target[..., l:] # ...predicts token n + 1

				if compute_hard_loss:
					nll = F.cross_entropy( logit, target, ignore_index=self.ignore_index )
					if 'nll' not in loss:
						loss['nll'] = []
					loss["nll"].append( nll )

				if compute_acc:
					if self.metrics is not None:
						metrics = self.metrics.calc_accuracy( [ logit ], [ target ], self.classifiers.indices([ classifier_level ]) )
					else:
						metrics = self.accuracy_metric( logit, target )

					if 'acc' not in stats:
						stats['acc'] = []
					stats["acc"].append( metrics )

		# average
		loss = { name: sum( loss[name] ) / len( loss[name] ) for name in loss.keys() }
		stats = { name: sum( stats[name] ) / len( stats[name] ) for name in stats.keys() }

		return LossStats(loss, stats)

	def forward(
		self,
		inputs: list,

		quant_levels: list[int] | None = None,
		state: dict | list | None = None,
		
		layer_skip_variables: dict | None = None,

		output_attentions: bool = False,
		output_hidden_states: bool = False,

		teacher = None,
	):
		# return early if it's "good" enough"
		# lambda because we need to capture the classifier_levels and mask
		exited_layer = self.n_layers
		def layer_skip_lambda( layer, logits ):
			nonlocal exited_layer
			kwargs = {
				"entropy_threshold": 0.05,
				"varentropy_threshold": 0.05,
				"min_layer": self.n_layers // 2,
				"max_layer": self.n_layers,
			}

			kwargs.update( layer_skip_variables )

			# don't bother on early layers
			if layer < kwargs["min_layer"]:
				return False
			# bail if we want to force early layers
			if kwargs["max_layer"] < layer:
				return True

			# hidden states aren't normalized
			x = self.model.norm( logits )

			# output projection layer with masking
			if self.classifier is not None:
				x = self.classifier(x) # * m
			elif self.classifiers is not None:
				logits = self.classifiers(logits, levels = classifier_levels) # * m

			# calculate metrics
			metrics = calculate_entropix_metrics( logits )
			# exit early if "good enough""
			early = metrics["logits_entropy"] <= kwargs["entropy_threshold"] and metrics["logits_varentropy"] <= kwargs["varentropy_threshold"]
			
			if early:
				exited_layer = layer

			#print( layer, early, metrics )

			return early

		# derive quant levels from inputs if not provided
		if quant_levels is None:
			quant_levels = self.get_input( inputs, "quant_level" )

		x_list = self.inputs_to_embeddings( inputs, quant_levels )
		
		x, mask = list_to_tensor(x_list)

		training = self.training
		teaching = self.teaching
		device = x.device
		batch_size = len(x_list)

		# pure AR
		if quant_levels is None:
			quant_levels = [ 0 for _ in range(batch_size) ]

		# we only need hidden states if we're training with layerskip
		if self.layerskip and training:
			output_hidden_states = True
		
		# pad our input and mask, but retain the original length by doing it after
		if self.l_padding and x.shape[1] % self.l_padding != 0:
			# pad input
			shape = list(x.shape)
			shape[1] = self.l_padding - shape[1] % self.l_padding

			padding = torch.zeros(shape, dtype=x.dtype, device=x.device)
			x = torch.cat([x, padding], dim=1)

			# pad mask
			shape[2] = 1
			padding = torch.zeros(shape[:2], dtype=x.dtype, device=x.device)
			mask = torch.cat([mask, padding], dim=1)
		
		m = mask.unsqueeze(dim=-1)

		# needs to be done here as we still have our raw inputs
		position_ids = self.inputs_to_position_ids( inputs, mask=mask ) if not self.unified_position_ids else None
		classifier_levels = self.get_input( inputs, name="classifier_level" )
		casual_levels = [ "AR:0:0", "stt", "len" ]

		# right now limit to new versions because I need to retrain the model for noncausal masks...
		is_causal = [ l in casual_levels for l in classifier_levels ] if self.noncausal_masks else None

		output = self._forward(
			inputs=x,
			mask=mask,
			state=state,
			is_causal=is_causal,
			position_ids=position_ids,
			output_attentions = output_attentions,
			output_hidden_states = output_hidden_states,
			layer_skip_lambda = layer_skip_lambda if self.layerskip and layer_skip_variables else None,
		)

		logits = output.logits
		hidden_states = output.hidden_states

		# output projection layer
		# the very, very original implementation multiplied by the mask, but the mask only attends to padding, and the padding gets removed anyways
		if self.classifier is not None:
			logits = self.classifier(logits) # * m
			
			if output.hidden_states:
				for i, state in enumerate( hidden_states ):
					hidden_states[i] = self.classifier(hidden_states[i]) # * m
		# to-do: piece-wise classification, now that there's a head for text
		# although again, one single monolithic head would be preferable instead......
		elif self.classifiers is not None:
			logits = self.classifiers(logits, levels = classifier_levels) # * m

			if hidden_states is not None:
				for i, state in enumerate( hidden_states ):
					hidden_states[i] = self.classifiers(hidden_states[i], levels = classifier_levels) # * m

		# Remove padding
		logits = [ hi[:li] for hi, li in zip(logits, map(len, x_list)) ]

		if hidden_states is not None:
			for i, state in enumerate( hidden_states ):
				hidden_states[i] = [ hi[:li] for hi, li in zip(hidden_states[i], map(len, x_list)) ]
		
		# compute loss if the target is given
		if not training:
			loss = None
			stats = None

			self.loss = None
			self.stats = None
		else:
			loss, stats = self.calc_loss( inputs=inputs, logits=logits, quant_levels=quant_levels )

			# compute it as an aux-loss
			if self.layerskip:
				early_exit_loss = {}
				if not hasattr( self, "training_steps" ):
					self.training_steps = 0
				
				for i, state in enumerate( hidden_states ):
					loss, stats = self.calc_loss( inputs=inputs, logits=hidden_states[i], quant_levels=quant_levels )
					
					for k, v in loss.items():
						K = f'early_exit.{k}'
						if K not in early_exit_loss:
							early_exit_loss[K] = []
						early_exit_loss[K].append( v )

				for k, v in early_exit_loss.items():
					loss[k] = self.model.early_exit_loss( losses=v, t=self.training_steps )

				# to-do: instead make the cirriculum rely on samples processed instead of steps
				self.training_steps += 1 # batch_size

			# get soft targets from teacher
			# required to do it in here because the batch is further processed within the model (because of per-model config)
			if teacher is not None:
				# grab the teacher's logits
				with torch.no_grad():
					teacher_output = teacher.forward_super(
						inputs=inputs,
						quant_levels=quant_levels,
					)

				# determine the output length for each batch (because blah blah some embeddings don't map to a discrete token anyways)
				# we could recreate the target sequence with the ignore indices put in, but that's agony
				output_lens = [ 0 for _ in range(batch_size) ]
				for batch_index, batch in enumerate(inputs):
					task_type = "tts"
					for name, input in batch:
						if name == "task":
							task_type = input

					for name, input in batch:
						if name == task_outputs.get(task_type, name):
							output_lens[batch_index] = input.shape[0]

				# KD hyperparameters
				T = self.teacher_temperature
				A = self.teacher_alpha

				# create probability distributions (literature says to have the students already log'd but not the teacher)
				student_probs = [ F.log_softmax( student[-l:] / T, dim=-1 ) for student, l in zip( logits, output_lens ) ]
				teacher_probs = [ F.softmax( teacher[-l:] / T, dim=-1 ) for teacher, l in zip( teacher_output.logits, output_lens ) ]

				# filter out logits that are / would inf
				# this causes problems when computing the loss if there's any inherently never-ever probabilities (for example, NAR RVQ-0 demasking for the stop token, because I did not clip it from the classifier)
				for batch_index, output_len in enumerate( output_lens ):
					mask_a = student_probs[batch_index] == -float("inf") # log(0) = -inf
					mask_b = teacher_probs[batch_index] == 0.0 # this gets log'd, eventually creating -inf

					mask = mask_a | mask_b
					student_probs[batch_index] = torch.masked_select( student_probs[batch_index], ~mask )
					teacher_probs[batch_index] = torch.masked_select( teacher_probs[batch_index], ~mask )

				#soft_losses = [ F.kl_div( student, teacher, reduction='mean' ) for student, teacher in zip( student_probs, teacher_probs ) ]
				#soft_losses = [ torch.sum(teacher * (teacher.log() - student)) for student, teacher in zip( student_probs, teacher_probs ) ]
				soft_losses = [ F.mse_loss( student, teacher ) for student, teacher in zip( student_probs, teacher_probs ) ]
				soft_loss = torch.stack([*soft_losses]).sum() * (T ** 2) / batch_size

				"""
				# flatten to a single sequence of token-probabilities
				# but this shouldn't actually work because some logits might be (..., 1024) and some might be (..., 1025)
				student_probs = torch.concat( student_probs, dim = 0 )
				teacher_probs = torch.concat( teacher_probs, dim = 0 )
				soft_loss = F.mse_loss( student_probs, teacher_probs ) * (T ** 2) / batch_size
				"""

				# mix if not nan
				if not torch.isnan(soft_loss).any():
					loss['kl'] = soft_loss * A
					for k in loss.keys():
						loss[k] *= (1.0 - A)

			# include any additional losses (for example: MoE router)
			if output.loss is not None:
				loss["aux_loss"] = output.loss

			self.loss = loss
			self.stats = stats
			
		# rewrap, because we're modifying the logits here
		return Logits(logits, output.state, loss, output.attentions, hidden_states, exited_layer)

	def sample(
		self,
		logits: list[Tensor], # logit scores
		prev_list: list[Tensor] | None = None, # previous tokens
		quant_levels: list[int] | None = None, # to-do: derive this from the prev_list
		**sampling_kwargs,
	):
		# yikes
		temperature = sampling_kwargs.get("temperature", 1.0)
		min_temperature = sampling_kwargs.get("min_temperature", -1.0)
		top_k = sampling_kwargs.get("top_k", -100)
		top_p = sampling_kwargs.get("top_p", 1.0)
		min_p = sampling_kwargs.get("min_p", 0.0)
		# repetition penalty parameters
		repetition_penalty = sampling_kwargs.get("repetition_penalty", 1.0)
		repetition_penalty_decay = sampling_kwargs.get("repetition_penalty_decay", 0.0)
		# length penalty parameters
		length_penalty = sampling_kwargs.get("length_penalty", 0.0)
		# beam sampling parameters
		beam_width = sampling_kwargs.get("beam_width", 0)
		# mirostat sampling parameters
		mirostat = sampling_kwargs.get("mirostat", None)
		# DRY sampling parameters
		dry_multiplier = sampling_kwargs.get("dry_multiplier", 0.0)
		dry_base = sampling_kwargs.get("dry_base", 1.75)
		dry_allowed_length = sampling_kwargs.get("dry_allowed_length", 2)
		#
		top_no = sampling_kwargs.get("top_no", 1.0)
		#
		attentions = sampling_kwargs.get("attentions", None)

		batch_size = len( logits )

		if min_temperature < 0:
			min_temperature = temperature

		# pick last RVQ level
		if prev_list is not None:
			prev_list = [ prevs if prevs.dim() == 1 else prevs[:, -1] for prevs in prev_list ]

		scores = None
		entropy = None
		#logits = [ logit.to(device="cpu", dtype=logit.dtype if logit.dtype != torch.float16 else torch.float32) for logit in logits ]
		#logits = [ logit.to(device="cpu") for logit in logits ]

		# (AR) entropix sampling
		# we do it before everything to retain logits for the entire sequence (even though it's still better to pass only the last token)
		if attentions is not None and quant_levels is None:
			# move to CPU for speedups
			seq_lens = [ logit.shape[0] for logit in logits ]
			attentions = torch.stack(attentions, dim=1).to(device="cpu") # ( batch, layer, heads, seq_len, seq_len )
			
			res = [ sample_entropix(
				logit[:seq_lens[batch], :], # ( seq_len, vocab )
				attentions[batch, :, :, :seq_lens[batch], :seq_lens[batch]], # (layer, heads, seq_len, seq_len )
				temperature,
				top_k,
				top_p,
				min_p,
			) for batch, logit in enumerate(logits) ]

			if res:
				return Sampled([ r[0] for r in res ], logits, scores, [ r[1] for r in res ])
		"""
		elif quant_levels is None:
			seq_lens = [ logit.shape[0] for logit in logits ]
			entropy = [ calculate_entropix_metrics(
				logit[:seq_lens[batch], :], # ( seq_len, vocab )
				#attentions[batch, :, :, :seq_lens[batch], :seq_lens[batch]], # (layer, heads, seq_len, seq_len )
			) for batch, logit in enumerate(logits) ]
		"""

		# (NAR) return the entire generated response
		# Parallel decoding relies on the last N tokens in the logits, because each token predicts the next RVQ layer in the same place (forgetfully obviously)		
		if quant_levels is not None: #  and "nar" in self.capabilities: # for when I get around to coping about dropping the NAR entirely
			seq_lens = map(len, prev_list)
			logits = [ logit[-l:] for logit, l in zip(logits, seq_lens) ]
		# (AR chunkwise) return the last chunkwise piece
		elif self.causal:
			seq_lens = [ logit.shape[0] - self.causal_size for logit in logits ]
			logits = [ logit[-self.causal_size:] for logit in logits ]

		# (NAR) disable stop token
		if quant_levels is not None and "ar" in self.capabilities:
			logits = [ ban_tokens(logit, tokens=[self.stop_token]) for logit, l in zip( logits, map(len, prev_list) ) ]
		# (AR-len) disable extraneous tokens
		"""
		if quant_levels is None and "len" in self.capabilities:
			logits = [ ban_tokens(logit, tokens=[*range(11, logit.shape[-1])]) for logit, l in zip( logits, map(len, prev_list) ) ]
		"""

		# perform repetition penalizing	
		if prev_list is not None and repetition_penalty != 1.0:
			logits = [ reptition_penalize(logit, previous=prevs, factor=repetition_penalty, decay=repetition_penalty_decay) for logit, prevs in zip( logits, prev_list ) ]

		# (AR) perform length penalizing
		if quant_levels is None and self.causal and prev_list is not None and length_penalty != 0.0:
			logits = [ length_penalize(logit, length=l + 1, factor=length_penalty, token=self.stop_token) for logit, l in zip( logits, map(len, prev_list) ) ]

		# perform min_p filtering of our logits
		if min_p > 0.0:
			logits = [ min_p_filtering(logit, min_p=min_p) for logit in logits ]

		# perform top_k/top_p filtering of our logits
		if top_k > 0 or top_p < 1.0:
			logits = [ top_k_top_p_filtering(logit, top_k=top_k, top_p=top_p) for logit in logits ]	

		# trigger dynamic temperature sampling if the minimum temperature is not the same as the sampling temperature
		#	 epsilon float comparison because I don't trust Python
		if abs(temperature - min_temperature) >= 0.001: 
			logits = [ dynamic_temperature(logit, temperature=temperature, min_temperature=min_temperature) for logit in logits ]
		elif temperature > 0.0:
			logits = [ logit / temperature for logit in logits ]

		# do top-no logit processing
		if top_no > 0.0:
			logits = [ top_no_logits_processing(logit) for logit in logits ]

		# do DRY sampling
		if dry_multiplier > 0.0 and prev_list is not None:
			logits = [ dry_sampling(logit, previous=prevs, factor=dry_multiplier, base=dry_base, allowed_length=dry_allowed_length) for logit, prevs in zip( logits, prev_list ) ]

		# do mirostat sampling
		# currently incompatible with beam searching with the way the two are implemented, perhaps a night of brain bashing can make the two work
		if mirostat is not None:
			# mirostat sampling
			scores = [ mirostat_sample(logit, state=state) for logit, state in zip(logits, mirostat) ]
			res = [ state["token"] for state in scores ]
		# do beam search (naive implementation)
		# picks the top-k across all batches, and re-batches those resultant tokens
		# returns the logit scores as well to be P-concatted with the previous scores
		# to-do: not naively implement beam searching
		elif beam_width > 1:
			candidates = top_k_logits_list( logits, beam_width )
			res = [ torch.tensor(token, dtype=torch.int16).unsqueeze(dim=-1) for batch, token in candidates ]
			scores = [ logits[batch].flatten()[token] for batch, token in candidates ]
		# basic sampling
		else:
			# argmax instead
			if temperature <= 0.0:
				res = [ logit.argmax(dim=-1) for logit in logits ]
			else:
				res = [ Categorical(logits=logit).sample() for logit in logits ]

			# calculate token probabilities
			scores = [
				[ F.softmax(logit[i, :], dim=-1)[token].item() for i, token in enumerate(tokens) ]
				for logit, tokens in zip(logits, res)
			]

		return Sampled(res, logits, scores, entropy)