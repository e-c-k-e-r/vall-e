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
Logits = namedtuple('Logits', ['logits', 'state', 'inputs', 'loss', 'attentions', 'hidden_states', 'exited_layer'])
Sampled = namedtuple('Sampled', ['ids', 'logits', 'scores', 'entropy'])
LossStats = namedtuple('LossStats', ['loss', 'stats'])

"""
from ..utils.pattern import DelayedPatternProvider, VALLEPattern
"""

summed_embeddings_task = [ "stt" ]
special_tasks = [ "len", "stt", "phn", "un-phn" ]
non_tokened_names = ["task", "dropout_mask", "classifier_level"]
task_outputs = {
	"tts": "resp",
	"ns": "resp",
	"sr": "resp",
	"stt": "text",
	"len": "len",
	"phn": "text",
	"un-phn": "raw_text",
}

# yuck
def _get_offsets():
	return {
		"text": (0, 256), 
		"quant_level": (256, 264), 
		"lang": (264, 270), 
		"task": (270, 279), 
		"len": (279, 290), 
		"tone": (290, 291), 
		"sep": (291, 292), 
		"prom|0": (292, 1316), 
		"prom|1": (1316, 2340), 
		"prom|2": (2340, 3364), 
		"prom|3": (3364, 4388), 
		"prom|4": (4388, 5412), 
		"prom|5": (5412, 6436), 
		"prom|6": (6436, 7460), 
		"prom|7": (7460, 8484), 
		"resps|AR:0:0": (8484, 9509), 
		"resps|NAR:0:1": (9509, 10533), 
		"resps|NAR:1:2": (10533, 11557), 
		"resps|NAR:2:3": (11557, 12581), 
		"resps|NAR:3:4": (12581, 13605), 
		"resps|NAR:4:5": (13605, 14629), 
		"resps|NAR:5:6": (14629, 15653), 
		"resps|NAR:6:7": (15653, 16677), 
		"resps|NAR:0:0": (16677, 17702), 
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
		l_embedding_tokens: int, # list of number of tokens (needed because AR resps includes stop token)
		token_dim: int, # dimensionality of the embedding
		levels: int | None = None, # number of RVQ-bins (I don't remember the specifics)
	):
		super().__init__()
		# array of embeddings
		#   proms are [0, resp_levels]
		#   resp are split to where [0] is for the AR, and [1:] are reserved for NAR
		self.embeddings = nn.ModuleList([nn.Embedding(n_tokens, token_dim) for n_tokens in l_embedding_tokens])
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
		l_embedding_tokens: list[int], # list of number of tokens (needed because AR resps includes stop token)
		token_dim: int, # dimensionality of the embedding
		sums: bool = True, # whether to sum all previous layers of embeddings to factor in other RVQ bin levels (I do not know which way is better)
		l_embedding_names: list[str] = [], # names to map to indices
	):
		super().__init__()
		# array of embeddings
		#   proms are [0, resp_levels]
		#   resp are split to where [0] is for the AR, and [1:] are reserved for NAR
		self.embeddings = nn.ModuleList([nn.Embedding(n_tokens, token_dim) for n_tokens in l_embedding_tokens])
		# further experimentation is needed to see if this actually is useful
		self.sums = sums
		# 
		self.names = l_embedding_names

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
		l_embedding_tokens: list[int], # list of number of tokens (needed because AR resps includes stop token)
		token_dim: int, # dimensionality of the embedding
		l_embedding_names: list[str] | None = None, # list of names to map to each classifier,
		bias: bool = True,
	):
		super().__init__()
		self.proj = nn.ModuleList([nn.Linear(token_dim, n_tokens, bias=bias) for n_tokens in l_embedding_tokens])
		self.names = l_embedding_names

	def indices(
		self,
		names
	):
		if isinstance( names[-1], int ):
			return names
		return [ self.names.index(name) for name in names ]

	def forward(self, xi: Tensor, levels: list[int] | None = None, names: list[str] | None = None, stack = False ) -> Tensor:
		dtype = xi.dtype
		device = xi.device

		if levels and isinstance( levels[-1], str ):
			names = levels
			levels = []

		# map names to levels
		if names and not levels:
			levels = [ self.names.index(name) for name in names ]

		xi = [ self.proj[l]( x ) for x, l in zip(xi, levels) ]
		if not stack:
			return xi

		# pad if needed
		# to-do: validate that this causes ZERO issues
		# addendum: this does cause problems
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
		l_embedding_tokens: int | list[int],
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
		) for n_tokens in l_embedding_tokens ])
		self.precision = nn.ModuleList([ MulticlassPrecision(
			n_tokens,
			top_k=top_k,
			average=average,
			multidim_average=multidim_average,
			ignore_index=ignore_index,
		) for n_tokens in l_embedding_tokens ])

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
		n_raw_text_tokens: int = 8575,

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
		self.n_raw_text_tokens = n_raw_text_tokens

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
		classifiers_bias = self.config.experimental.classifiers_bias if self.config is not None else False
		max_position_embeddings = self.config.experimental.max_position_embeddings if self.config is not None else (75 * 60 * 5)
		
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
			l_embedding_tokens = [n_resp_tokens] * self.n_resp_levels
			l_embedding_names = [f'AR:{i}:{i}' for i in range( self.n_resp_levels )]
			l_classifier_tokens = [n_resp_tokens] * self.n_resp_levels
		# NAR-len model
		elif "len" in self.capabilities:
			# +1 to include the stop or mask token
			n_resp_tokens = n_audio_tokens + ( 1 if self.causal_size > 0 else 0 )
			if "ar" in self.capabilities:
				l_embedding_tokens = [n_resp_tokens] + [n_resp_tokens - 1] * (self.n_resp_levels - 1) + [n_resp_tokens]
				l_classifier_tokens = [n_resp_tokens] + [n_resp_tokens - 1] * (self.n_resp_levels - 1) + [n_resp_tokens - 1]
				l_embedding_names = ['AR:0:0'] + [f'NAR:{i}:{i+1}' for i in range( self.n_resp_levels - 1 )] + ['NAR:0:0']
			else:
				l_embedding_tokens = [n_resp_tokens] + [n_resp_tokens - 1] * (self.n_resp_levels - 1)
				l_classifier_tokens = [n_resp_tokens] + [n_resp_tokens - 1] * (self.n_resp_levels - 1)
				l_embedding_names = ['NAR:0:0'] + [f'NAR:{i}:{i+1}' for i in range( self.n_resp_levels - 1 )]
		# AR+NAR model
		else:
			# +1 to include the stop or mask token
			n_resp_tokens = n_audio_tokens + ( 1 if self.causal_size > 0 else 0 )
			l_embedding_tokens = [n_resp_tokens] + [n_resp_tokens - 1] * (self.n_resp_levels - 1)
			l_embedding_names = ['AR:0:0'] + [f'NAR:{i}:{i+1}' for i in range( self.n_resp_levels - 1 )]
			l_classifier_tokens = [n_resp_tokens] + [n_resp_tokens - 1] * (self.n_resp_levels - 1)
		
		l_classifier_names = l_embedding_names

		# STT
		l_classifier_names += [ "stt" ]
		l_classifier_tokens += [ n_text_tokens ]

		# LEN
		if "len" in self.capabilities:
			l_classifier_tokens += [ 11 ]
			l_classifier_names += ["len"]

		# TEXT => PHN / PHN => TEXT
		if self.version >= 6:
			l_classifier_tokens += [ n_raw_text_tokens ]
			l_classifier_names = l_embedding_names + [ "raw_text" ]

		n_vocab = 17702 if not split_classifiers else n_resp_tokens + 1

		self.n_vocab = n_vocab
		self.unified_position_ids = unified_position_ids
		self.interleave = interleave
		self.layerskip = layerskip
		self.inject_timestep_embedding = False # results in bad output
		self.masking_ratio = masking_ratio
		self.ignore_inputs_for_loss = ignore_inputs_for_loss
		self.noncausal_masks = noncausal_masks

		# use internal attention mechanism for now because I dont have a better way to handle mixed causal/noncausal masks for other attention backends
		"""
		if noncausal_masks:
			attention_backend = "default"
		"""

		self.text_emb = Embedding(n_text_tokens, d_model)
		self.raw_text_emb = None
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
				l_embedding_tokens, d_model,
				levels=self.n_resp_levels if self.version > 3 else None,
			)
		else:
			self.proms_emb = AudioEmbedding(
				[n_audio_tokens] * self.n_resp_levels, d_model,
				sums=audio_embedding_sums == "prom" or audio_embedding_sums == True,
			)
			self.resps_emb = AudioEmbedding(
				l_embedding_tokens, d_model,
				sums=audio_embedding_sums == "resp" or audio_embedding_sums == True,
				l_embedding_names=l_embedding_names,
			)

		if self.version >= 3:
			self.langs_emb = Embedding(n_langs, d_model) if n_langs > 0 else None
			self.tasks_emb = Embedding(n_tasks, d_model) if n_tasks > 0 else None

			self.capabilities += ["lang"]
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

		if self.version >= 6:
			self.raw_text_emb = Embedding(self.n_raw_text_tokens, d_model)

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
					vocab_size=n_vocab,
					hidden_size=d_model,
					max_position_embeddings=max_position_embeddings,
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
					max_position_embeddings=max_position_embeddings,
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
				config = LlamaConfig(
					vocab_size=n_vocab,
					hidden_size=d_model,
					max_position_embeddings=max_position_embeddings,
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
				)
				self.model = LlamaClass(config)

				# replace with desired attention
				if attention_backend not in HF_ATTENTIONS:
					self.model = ml.replace_attention( self.model, klass=LlamaAttention_Adapted, target=LlamaAttention, mode=attention_backend )
			else:
				self.model = MixtralModel(MixtralConfig(
					vocab_size =n_resp_tokens,
					hidden_size=d_model,
					max_position_embeddings=max_position_embeddings,
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
				vocab_size=n_vocab,
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
				vocab_size=n_vocab,
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
				vocab_size=n_vocab,
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
			self.classifier = nn.Linear(d_model, n_vocab, bias=classifiers_bias)
			self.classifiers = None

			self.metrics = None
		else:
			self.classifier = None
			self.classifiers = Classifiers( l_classifier_tokens, d_model, l_embedding_names=l_classifier_names, bias=classifiers_bias )
			self.metrics = Metrics( l_classifier_tokens )

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

		return Logits(x, state, inputs, aux_loss, attentions, hidden_states, None)

	# takes a bunch of separate lists and parses them into an ordered array of tuples to guide input sequence creation
	def inputs(
		self,
		text_list: list[Tensor] | None = None,
		raw_text_list: list[Tensor] | None = None,

		proms_list: list[Tensor] | None = None,
		resps_list: list[Tensor] | None = None,

		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
		len_list: list[Tensor] | None = None,
		task_list: list[str] | None = None,
		time_list: list[Tensor] | None = None,

		quant_levels: int | list[int] | Tensor | None = None
	):
		if text_list and text_list[0] is not None:
			device = text_list[0].device
			batch_size = len(text_list)
		elif raw_text_list and raw_text_list[0] is not None:
			device = raw_text_list[0].device
			batch_size = len(raw_text_list)
		elif proms_list and proms_list[0] is not None:
			device = proms_list[0].device
			batch_size = len(proms_list)
		elif resps_list and resps_list[0] is not None:
			device = resps_list[0].device
			batch_size = len(resps_list)

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
			if task_type in get_task_symmap() and task_type not in special_tasks:
				# insert the text prompt
				if text_list is not None and text_list[i] is not None:
					inputs[i].append( ( "text", text_list[i] ) )
				elif raw_text_list is not None and raw_text_list[i] is not None:
					inputs[i].append( ( "raw_text", raw_text_list[i] ) )
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
				elif raw_text_list is not None and raw_text_list[i] is not None:
					inputs[i].append( ( "raw_text", raw_text_list[i] ) )
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
			# Text phonemizing task
			# Sequence: <raw_text><sep><lang><sep><phonemes>
			elif task_type == "phn":
				# insert the text prompt
				if raw_text_list is not None and raw_text_list[i] is not None:
					inputs[i].append( ( "raw_text", raw_text_list[i] ) )
				# insert lang token if we're trained for it
				if "lang" in self.capabilities and lang_list is not None and lang_list[i] is not None:
					inputs[i].append( ( "lang", lang_list[i] ) )
				if self.rvq_l_emb is not None and not self.interleave:
					inputs[i].append( ( "quant_level", torch.tensor([ quant_level ], device=device, dtype=torch.int16) ) )
				# insert the text prompt
				if text_list is not None and text_list[i] is not None:
					inputs[i].append( ( "text", text_list[i] ) )

				inputs[i].append( ("classifier_level", "stt") )
			# Text de-phonemizing task
			# Sequence: <raw_text><sep><lang><sep><phonemes>
			elif task_type == "un-phn":
				# insert the text prompt
				if text_list is not None and text_list[i] is not None:
					inputs[i].append( ( "text", text_list[i] ) )
				# insert lang token if we're trained for it
				if "lang" in self.capabilities and lang_list is not None and lang_list[i] is not None:
					inputs[i].append( ( "lang", lang_list[i] ) )
				if self.rvq_l_emb is not None and not self.interleave:
					inputs[i].append( ( "quant_level", torch.tensor([ quant_level ], device=device, dtype=torch.int16) ) )
				# insert the text prompt
				if raw_text_list is not None and raw_text_list[i] is not None:
					inputs[i].append( ( "raw_text", raw_text_list[i] ) )
				
				inputs[i].append( ("classifier_level", "raw_text") )
			else:
				raise Exception(f'Unrecognized task: {task_type}')
		return inputs

	def offset_inputs(
		self,
		inputs: list,
		direction: int = 1, # -1 to de-offset
	):
		offsets = _get_offsets()

		for batch_index, batch_input in enumerate(inputs):
			quant_level = None
			classifier_level = None
			# pre-iterate
			for name, input in batch_input:
				if name == "quant_level":
					quant_level = input
				elif name == "classifier_level":
					classifier_level = input

			for name, input in batch_input:
				if not isinstance( input, torch.Tensor ):
					continue

				k = name
				if name == "prom":
					k = f'prom|{quant_level}'
				elif name == "resp":
					k = f'resps|{classifier_level}'

				if k not in offsets:
					continue

				start, end = offsets[k]

				for i, t in enumerate( input ):
					input[i] += start * direction

		return inputs

	def inputs_to_embeddings(
		self,
		inputs: list,
		quant_levels: int | list[int] | Tensor | None = None
	):
		# handles tasks where the prompt has task tokens injected in the middle
		def prompt_input_to_embedding( input, quant_level ):
			if isinstance(input, str):
				return self.tasks_emb( torch.tensor( [ get_task_symmap()[input] ], device=device, dtype=torch.int16) )

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
				elif name == "raw_text" and self.raw_text_emb is not None:
					embedding = self.raw_text_emb( input )

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
			if isinstance(input, str):
				return torch.tensor( [ get_task_symmap()[input] ], device=device, dtype=torch.int16)

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

				# offset to flattened vocab ranges
				"""
				if self.classifier is not None:
					offsets = _get_offsets()

					k = name
					if name == "stt":
						k = "text"
					if name == "prom":
						k = f'prom|{quant_level}'
					elif name == "resp":
						k = f'resps|{classifier_level}'
					
					if k in offsets:
						start, end = offsets[k]

						for i, t in enumerate( token ):
							if t == self.ignore_index:
								continue
							token[i] += start
				"""

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
					token = torch.tensor( [ self.ignore_index ] * token.shape[0], device=device, dtype=torch.int16)
					
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
							accuracy_metric = MulticlassAccuracy(
								logit.shape[-1],
								top_k = 10,
								average="micro",
								multidim_average="global",
								ignore_index = -100
							).to(logit.device)
							metrics = accuracy_metric( logit, token )

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
						accuracy_metric = MulticlassAccuracy(
							logit.shape[-1],
							top_k = 10,
							average="micro",
							multidim_average="global",
							ignore_index = -100
						).to(logit.device)
						metrics = accuracy_metric( logit, target )

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

			return early

		# derive quant levels from inputs if not provided
		if quant_levels is None:
			quant_levels = [ x.item() for x in self.get_input( inputs, "quant_level" ) ]

		# inputs don't have quant levels added, pure AR
		if len(quant_levels) != len(inputs):
			quant_levels = [ 0 for _ in range(len(inputs)) ]

		x_list = self.inputs_to_embeddings( inputs, quant_levels )
		
		x, mask = list_to_tensor(x_list)

		training = self.training
		teaching = self.teaching
		device = x.device
		batch_size = len(x_list)

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
		casual_levels = [ "AR:0:0", "stt", "len", "phn" ]

		# right now limit to new versions because I need to retrain the model for noncausal masks...
		is_causal = [ l in casual_levels for l in classifier_levels ] if self.noncausal_masks else [ True for l in classifier_levels ]

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
		
		# de-offset if needed
		if self.classifier is not None:
			offsets = _get_offsets()
			for batch_index, classifier_level in enumerate( classifier_levels ):
				if classifier_level == "stt":
					k = "text"
				elif classifier_level == "len":
					k = "len"
				else:
					k = f'resps|{classifier_level}'

				if k not in offsets:
					continue

				start, end = offsets[k]

				logits[batch_index] = logits[batch_index][:, start:end]

		if not training:
			loss = None
			stats = None

			self.loss = None
			self.stats = None
		# compute loss if the target is given
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

			# include any additional losses (for example: MoE router)
			if output.loss is not None:
				loss["aux_loss"] = output.loss

			self.loss = loss
			self.stats = stats
			
		# rewrap, because we're modifying the logits here
		return Logits(logits, output.state, inputs, loss, output.attentions, hidden_states, exited_layer)

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

# this is a VERY basic implementation to test if a HF-ified model works (it sort of does)
if __name__ == "__main__":
	from transformers import LlamaForCausalLM, LlamaTokenizer
	from ..models import download_model, DEFAULT_MODEL_PATH

	from ..emb.qnt import decode_to_file
	from ..utils.io import torch_load

	# hack in a non-causal mask
	def _update_noncausal_mask(
		attention_mask,
		inputs_embeds,
		cache_positions,
		past_key_values_length,
		output_attentions,
	):
		# create noncausal mask
		# [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]

		bsz, seq_len, _ = inputs_embeds.size()

		# generate default mask based on input
		if attention_mask is None:
			attention_mask = torch.ones( (bsz, seq_len), dtype=torch.bool, device=inputs_embeds.device )

		# make square
		expanded_mask = attention_mask[:, None, None, :].expand( bsz, 1, seq_len, seq_len ).to( dtype=inputs_embeds.dtype )

		# invert from 1.0 = attend, 0.0 = masked to 0.0 = valid, -inf = masked
		inverted_mask = 1.0 - expanded_mask
		return inverted_mask.masked_fill( inverted_mask.to(dtype=torch.bool), torch.finfo(inputs_embeds.dtype).min )

	device = "cuda"
	dtype = torch.bfloat16

	is_from_pretrained = True
	if is_from_pretrained:
		# tokenizer = LlamaTokenizer.from_pretrained("ecker/vall-e", revision="hf")
		hf_model = LlamaForCausalLM.from_pretrained("ecker/vall-e", revision="hf")
		hf_model.to(device=device, dtype=dtype)
		hf_model.eval()

		model = hf_model.model
	else:
		download_model()
		model = LlamaModel(LlamaConfig(
			vocab_size=1024,
			hidden_size=1024,
			max_position_embeddings=75 * 60 * 5, # max-length of 60 seconds
			intermediate_size=1024*4,
			num_hidden_layers=12,
			num_attention_heads=16,
			attention_dropout=0.0,
			num_key_value_heads=16,
			sliding_window=75 * 12, # 12 second context window
			hidden_act="gelu",
			is_encoder_decoder=False,
			is_decoder=True,
		))

		state_dict = torch_load(DEFAULT_MODEL_PATH)['module']
		state_dict_model = {}
		for k, v in state_dict.items():
			if not k.startswith('model.'):
				continue
			state_dict_model[k.replace("model.", "")] = v

		model.load_state_dict( state_dict_model, strict=False )
		model.to(device=device, dtype=dtype)
		model.eval()

	model._original_update_causal_mask = model._update_causal_mask
	model._update_noncausal_mask = _update_noncausal_mask

	phn = [1,22,111,100,4,37,115,169,11,2]

	prom = [
		[62,835,835,835,339,395,798,537,537,537,537,222,76,989,548,65,705,375,261,375,297,503,529,571,707,346,266,862,148,496,574,115,115,438,934,339,865,876,63,40,779,461,602,794,10,220,507,869,639,705,869,917,705,893,917,705,869,938,439,175,139,506,375,529,297,705,651,238,962,461,195,441,377,581,473,795,644,626,459,981,767,670,696,73,779,257,738,1017,1019,133,133,1017,835,604,699,626,67,92,707,92,179,179,772,869,441,799,630,238,745,904,904,904,106,133,133,1017,1017,395,883,87,519,594,1002,682,996,540,186,855,430,202,347,889,61,92,542,297,67,669,571,707,346,67,359,571,707,669,604,395,1008,810,35,621,67,600,333,123,284,568,817,243,778,464,638,610,359,538,464,975,321,700,377,484,179,284,284,621,538,464,745,171,171,159,744,744,287,461,69,15,529,67,92,669,464,515,605,24,822,865,293,865,172,638,359,562,138,839,846,775,556,775,1006,917,346,312,148,331,496,646,67,314,15,705,131,855,662,287,172,85,107,519,374,450,391,609,643,778,80,287,794,794,115,785,794,461,699,519,932,522,652,262,508,902,932,932,391,769,18,507,90,442,762,610,610,669,605,35,855,56,989,863,195,464,604,257,904,632,786,951,461,239,195,878,771,146,481,146,481,434,643,917,280,67,464,115,744,744,115,115,115,819,709,63,907,359,519,996,616,682,996,616,519,762,917,841,772,568,954,600,422,893,592,464,626,86,143,615,171,744,744,196,115,821,415,521,799,654,839,644,473,592,953,523,855,738,855,855,876,1017,63,329],
		[913,859,740,740,937,601,961,961,877,747,747,559,474,618,20,316,58,316,180,112,290,869,610,869,869,943,127,153,236,794,282,857,984,196,875,648,993,913,860,616,38,833,620,133,123,992,247,367,252,50,298,27,27,631,163,784,271,20,843,514,869,258,180,66,803,281,123,493,831,102,556,992,385,122,31,251,990,827,26,347,460,43,43,460,228,43,841,913,302,544,544,277,859,404,646,775,315,848,726,185,203,314,203,174,252,174,378,954,214,993,924,809,277,765,363,544,363,518,791,185,454,193,193,193,193,193,573,977,924,76,434,56,193,962,610,24,954,459,396,112,903,137,398,474,506,791,839,399,102,25,205,792,459,474,526,817,869,192,792,593,878,506,24,410,539,788,522,667,566,584,588,992,444,24,869,925,635,393,903,742,320,1023,833,136,216,924,220,24,563,630,968,96,708,24,708,127,399,364,67,740,381,981,203,248,607,744,252,996,474,582,248,527,423,25,387,94,229,775,122,474,792,367,650,371,413,448,448,784,506,795,848,298,27,526,96,905,70,693,956,1002,1002,37,747,857,993,124,193,193,193,193,732,732,732,992,447,792,929,291,289,524,451,27,27,524,202,693,374,1002,125,732,585,367,317,679,395,413,189,493,386,650,110,912,505,384,399,851,367,367,27,230,988,810,975,842,956,1002,4,551,729,956,1002,750,648,231,950,193,96,912,410,732,539,103,193,904,491,213,792,792,998,193,399,151,410,96,673,497,1002,241,833,956,630,43,399,775,732,792,792,792,792,917,750,185,812,812,700,859,841,363,833,630],
		[786,36,821,937,1000,705,1016,345,345,470,165,581,95,404,95,95,1006,477,95,95,691,254,997,657,176,124,95,673,489,326,218,437,907,590,752,541,1016,821,445,563,181,555,181,345,576,190,987,0,265,997,488,12,598,687,152,108,52,95,95,71,87,945,95,997,754,488,955,694,925,82,18,1020,1006,542,788,441,325,532,246,132,560,532,947,655,653,842,732,36,36,829,36,937,989,989,752,651,87,489,677,260,789,462,95,227,986,955,95,810,624,435,280,868,832,879,863,821,829,937,168,270,489,544,909,562,957,0,593,714,675,690,626,227,794,489,489,563,489,298,269,741,249,516,360,240,516,336,93,808,1022,682,555,737,147,405,476,895,323,694,412,689,963,72,193,298,181,521,741,193,93,153,773,677,689,495,30,564,719,1020,559,940,53,53,53,929,360,971,403,1012,997,919,957,433,919,787,401,401,355,276,370,414,690,697,330,629,552,930,720,259,579,221,62,945,135,1020,626,663,401,153,997,381,830,185,587,853,207,126,66,529,410,113,997,488,431,563,488,488,719,746,790,296,843,752,790,23,984,292,41,27,120,249,124,900,358,801,227,978,95,997,997,997,371,561,86,388,52,667,601,894,545,997,498,900,494,365,852,986,95,841,664,256,18,1020,963,901,447,498,262,388,691,997,646,651,757,468,114,601,437,940,212,655,541,970,870,521,237,957,563,794,563,564,620,489,351,489,489,257,733,629,489,227,622,962,7,598,374,470,114,159,211,298,363,843,818,153,59,452,529,258,419,605,689,526,39,982,829,982,752,678,1005,312],
		[673,673,919,866,762,961,52,674,528,528,675,526,12,753,297,967,661,845,482,303,338,1021,506,445,247,214,206,94,434,799,210,885,552,695,853,1022,916,762,764,721,445,434,529,999,771,708,767,498,282,736,227,150,299,12,536,767,321,561,12,530,147,530,262,325,196,990,874,997,944,875,426,12,282,571,571,282,365,534,365,424,89,388,563,222,31,1019,624,74,215,651,1018,74,956,1022,74,18,633,350,72,448,454,769,267,938,12,534,929,723,829,614,505,364,1018,1014,838,673,919,74,223,761,266,78,177,736,20,718,425,1001,366,58,874,58,153,627,312,197,801,530,767,674,196,633,327,425,376,413,1019,209,594,383,744,458,468,711,282,885,640,435,655,571,556,1020,310,116,273,116,504,633,15,736,633,448,662,612,487,345,19,612,665,556,198,778,705,403,706,31,196,197,536,805,427,339,161,241,116,504,58,945,853,734,670,424,807,19,397,175,144,419,19,221,697,68,321,800,210,824,972,712,911,362,427,694,182,651,972,863,684,887,548,806,27,627,639,432,193,103,198,436,837,366,212,125,1001,493,874,808,17,17,127,204,530,300,345,425,246,240,640,906,340,310,633,246,774,114,633,522,777,874,494,577,353,939,571,693,857,722,530,521,354,492,735,214,806,483,736,530,118,234,536,177,132,522,349,259,436,973,528,414,224,762,212,854,744,271,568,127,323,736,304,499,499,78,536,736,805,232,126,468,566,611,52,339,450,258,157,602,594,854,602,599,82,124,472,563,666,174,936,818,66,758,627,52,350,999,734,215,919,1018,874,885],
		[528,448,646,190,222,884,939,907,907,673,413,786,527,517,710,449,119,531,565,762,531,501,522,246,162,871,8,594,206,937,462,712,862,151,103,261,882,990,1007,314,683,864,693,812,319,786,107,531,31,342,632,460,269,429,531,531,717,417,321,671,1015,152,467,863,285,875,941,417,475,825,596,957,117,460,162,162,117,630,735,527,272,558,38,39,605,375,39,900,862,646,712,804,622,963,407,93,828,796,306,415,70,667,371,531,1000,411,710,162,812,381,673,498,691,884,928,712,528,48,630,24,593,901,973,579,722,75,139,909,919,328,764,393,777,753,512,577,175,577,512,922,834,863,30,69,94,68,616,691,835,335,486,345,306,374,732,938,580,311,715,495,527,1008,306,369,663,512,369,320,360,80,42,1021,1021,1021,175,568,526,362,320,317,488,613,937,548,966,545,596,177,306,480,522,577,512,512,638,1008,82,100,696,89,714,531,639,460,679,718,492,509,492,624,460,572,531,306,19,473,915,558,285,319,713,1018,381,877,667,425,905,43,437,632,634,324,306,207,324,303,48,69,467,39,902,599,3,617,465,78,918,459,1009,427,751,145,531,349,356,1021,157,507,780,624,165,507,144,270,94,414,899,379,947,994,853,107,586,652,877,92,19,91,188,544,624,470,503,513,13,192,563,145,531,618,743,470,62,701,499,436,679,505,198,959,3,766,839,437,491,395,1021,512,306,512,356,851,1021,1021,78,690,856,735,286,280,4,1008,369,359,309,651,864,561,170,692,952,877,520,959,306,37,1021,31,236,162,773,522,254,446,606,691,804,882,58,974],
		[1011,939,881,881,140,937,724,724,937,1011,381,229,965,251,745,69,305,206,566,813,503,116,940,127,353,621,57,779,595,744,755,530,701,862,760,443,293,768,156,281,960,504,327,979,55,790,545,953,830,759,667,485,861,63,485,55,898,581,520,49,99,651,940,945,685,621,728,487,650,530,934,378,522,522,522,996,534,522,739,534,378,543,94,602,390,948,692,692,41,41,768,412,982,692,692,774,176,791,526,497,57,940,542,685,694,916,813,890,357,193,430,863,929,412,412,903,140,763,465,707,569,925,859,985,24,411,835,298,293,791,837,460,182,296,137,474,809,111,376,1021,111,490,111,938,542,578,477,506,57,385,300,873,240,104,667,204,515,834,24,125,113,980,111,997,859,997,376,193,490,824,511,799,719,575,451,575,251,222,630,429,920,788,300,993,641,154,816,940,618,130,940,462,823,955,1001,569,508,632,2,903,399,333,709,489,726,932,725,777,970,843,717,940,211,534,274,161,392,103,31,462,813,985,638,213,352,219,236,381,287,111,87,818,953,112,336,980,1016,72,960,426,238,60,9,487,665,129,24,24,162,312,411,111,157,473,466,222,940,341,55,457,712,179,451,111,831,918,826,814,940,30,468,240,207,389,923,186,95,300,876,679,576,543,582,111,227,312,112,545,747,378,165,158,610,601,425,238,704,630,124,644,949,982,297,868,569,24,57,465,24,859,111,24,752,775,24,647,465,495,57,24,57,227,907,296,581,843,1013,514,555,319,937,347,478,186,684,15,241,534,369,381,846,578,314,711,814,435,41,986,673,991],
		[485,748,562,562,485,380,834,997,78,963,755,142,978,135,362,421,217,79,530,1012,972,946,127,587,838,818,456,548,424,479,944,650,694,447,391,616,938,908,206,259,998,292,818,128,353,273,566,796,333,146,110,986,571,451,166,229,421,300,911,689,329,145,287,273,542,808,301,491,0,278,825,442,0,100,818,826,66,904,642,566,135,305,999,993,905,485,755,782,365,977,485,1015,570,1002,755,169,967,36,721,1019,273,931,273,166,216,31,346,946,32,290,362,828,464,748,782,1002,1015,755,1014,100,315,777,549,177,882,110,603,975,531,608,67,1011,950,465,368,416,798,941,635,602,553,300,200,644,498,325,786,734,342,222,403,1,716,175,899,273,40,333,999,74,54,644,408,976,407,631,577,338,435,612,333,273,162,709,882,555,384,995,173,459,442,72,72,200,72,711,219,282,716,442,431,801,976,130,622,72,582,384,516,772,0,440,1001,249,1,953,65,945,438,249,511,561,205,507,821,998,427,746,290,544,426,693,999,190,214,167,219,534,166,325,975,414,326,326,268,679,991,418,868,445,632,160,380,890,346,315,806,258,806,486,326,797,471,18,790,33,66,63,66,224,38,599,599,110,801,761,18,936,230,253,171,393,774,887,887,403,466,495,524,261,666,256,687,759,263,713,185,454,242,988,185,161,911,430,86,550,439,327,527,671,782,383,916,590,315,806,583,465,785,321,315,421,856,66,352,0,634,540,362,948,185,16,224,372,694,259,648,87,733,659,603,67,269,901,66,566,173,705,746,566,911,10,743,860,78,782,1002,755,389,175],
		[948,948,975,975,948,322,672,639,902,55,916,439,498,389,407,682,451,401,386,440,499,348,736,891,603,762,783,407,886,76,543,699,137,458,639,253,63,475,55,436,502,888,542,131,524,167,738,131,907,29,378,545,227,382,478,399,218,872,917,202,330,2,371,264,667,355,1016,768,590,408,463,542,214,202,715,891,840,297,509,689,290,439,672,714,528,940,1019,534,975,475,1019,835,975,558,975,981,330,635,96,858,606,627,367,191,191,669,40,873,359,267,701,426,210,1012,899,975,475,1012,610,6,300,749,231,616,877,631,720,574,551,398,503,789,684,664,390,277,150,990,823,190,971,903,175,863,316,965,988,988,800,612,336,506,242,847,389,939,415,202,83,317,2,153,365,363,57,2,891,965,300,754,763,426,555,621,303,415,367,902,829,741,119,380,902,25,884,439,822,49,76,760,566,316,249,555,774,955,834,309,859,173,935,812,682,586,141,606,197,131,644,631,913,586,202,117,810,884,76,592,754,531,586,925,649,583,145,816,821,283,871,1017,316,377,646,339,201,76,780,76,976,217,38,598,977,617,825,833,49,231,749,749,633,205,231,271,50,249,684,555,982,526,895,288,22,57,722,996,260,1018,110,833,644,738,648,468,798,297,769,282,197,402,465,510,194,930,182,909,749,986,187,187,917,38,38,985,985,988,815,878,814,459,237,768,781,649,683,749,934,729,463,181,625,231,917,96,499,839,720,439,842,205,808,338,617,681,326,446,905,346,647,533,49,728,147,432,846,536,586,611,49,879,872,893,859,859,961,989,975,701,495,65],
	]
	resp = []
	"""
	resp = [
		[922,738,461,341,341,10,416,416,416,416,346,346,346,346,346,484,484,484,484,484,484,333,442,442,359,359,359,459,459,975,975,626,626,626,626,626,610,359,359,359,359,359,359,359,359,359,610,610,442,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,638,638,638,638,975,975,672,875,63,144],
		[993,700,384,213,794,10,305,778,58,225,118,260,768,768,260,474,903,732,70,992,447,70,1000,665,848,379,485,934,181,795,438,298,688,324,934,756,395,795,110,328,343,172,768,871,593,355,396,783,24,24,911,20,27,562,697,616,668,27,27,755,20,505,248,79,822,461,197,156,27,492,151,1013,669,669,562],
		[626,989,936,488,511,624,997,112,112,648,210,650,563,650,41,41,490,920,977,986,920,927,131,167,167,968,346,168,167,168,120,355,766,599,712,390,558,810,948,332,332,867,994,346,955,392,920,452,576,346,52,254,52,307,897,307,968,920,167,563,167,167,167,968,167,488,968,488,1001,938,563,741,432,566,758],
		[916,874,798,212,496,751,620,616,982,745,975,890,890,141,141,321,321,214,899,42,151,722,310,971,774,35,627,995,27,43,248,248,595,774,942,352,810,35,384,340,654,639,89,214,737,197,657,45,622,321,337,19,483,679,938,938,682,938,938,141,938,310,114,724,116,327,372,607,607,310,204,713,762,853,853],
		[528,222,992,727,536,191,202,483,306,568,533,577,398,533,202,24,753,753,739,739,643,513,4,324,369,66,447,201,66,802,66,957,665,526,602,749,483,447,193,853,531,201,201,71,888,202,66,66,650,228,533,102,639,513,533,531,533,471,344,566,201,639,471,639,732,594,464,308,116,533,116,174,959,621,539],
		[692,632,478,375,910,857,775,503,503,193,717,548,344,717,55,808,162,112,112,112,543,582,847,712,691,679,427,940,369,475,153,526,729,269,323,721,526,211,191,192,685,844,731,813,914,545,582,712,925,916,375,111,340,162,844,940,844,162,844,990,111,491,232,582,491,582,618,121,1020,664,670,254,315,438,723],
		[365,908,896,819,206,153,515,471,75,79,664,145,145,801,135,321,79,216,233,223,79,66,724,517,135,474,818,818,105,892,971,337,818,19,932,981,469,135,163,75,135,818,999,555,135,710,256,105,590,31,539,1003,517,130,445,40,549,130,859,385,1003,1003,549,33,286,932,329,774,321,664,686,16,834,703,290],
		[899,237,832,748,425,121,460,872,391,586,857,215,306,76,306,554,187,57,482,406,802,555,710,895,448,517,506,316,18,772,779,697,855,1005,792,96,402,96,517,775,506,938,114,986,986,503,749,984,524,527,506,749,463,490,188,374,506,49,537,188,494,900,526,524,524,500,500,345,630,338,982,761,700,598,749],
	]
	"""

	# name, (start, end), classifier, src_name
	io_map = {
		'text': [(0, 256), 9, "text_emb.weight"],
		'rvq_l': [(256, 264), -1, "rvq_l_emb.weight"],
		'lang': [(264, 270), -1, "langs_emb.weight"],
		'task': [(270, 279), -1, "tasks_emb.weight"],
		'len': [(279, 290), 10, "len_emb.weight"],
		'tone': [(290, 291), -1, "tones_emb.weight"],
		'sep': [(291, 292), -1, "sep"],
		'prom|0': [(292, 1316), -1, "proms_emb.embeddings.0.weight"],
		'prom|1': [(1316, 2340), -1, "proms_emb.embeddings.1.weight"],
		'prom|2': [(2340, 3364), -1, "proms_emb.embeddings.2.weight"],
		'prom|3': [(3364, 4388), -1, "proms_emb.embeddings.3.weight"],
		'prom|4': [(4388, 5412), -1, "proms_emb.embeddings.4.weight"],
		'prom|5': [(5412, 6436), -1, "proms_emb.embeddings.5.weight"],
		'prom|6': [(6436, 7460), -1, "proms_emb.embeddings.6.weight"],
		'prom|7': [(7460, 8484), -1, "proms_emb.embeddings.7.weight"],
		'resp|AR:0:0': [(8484, 9509), 0, "resps_emb.embeddings.0.weight"],
		'resp|NAR:0:1': [(9509, 10533), 1, "resps_emb.embeddings.1.weight"],
		'resp|NAR:1:2': [(10533, 11557), 2, "resps_emb.embeddings.2.weight"],
		'resp|NAR:2:3': [(11557, 12581), 3, "resps_emb.embeddings.3.weight"],
		'resp|NAR:3:4': [(12581, 13605), 4, "resps_emb.embeddings.4.weight"],
		'resp|NAR:4:5': [(13605, 14629), 5, "resps_emb.embeddings.5.weight"],
		'resp|NAR:5:6': [(14629, 15653), 6, "resps_emb.embeddings.6.weight"],
		'resp|NAR:6:7': [(15653, 16677), 7, "resps_emb.embeddings.7.weight"],
		'resp|NAR:0:0': [(16677, 17702), 8, "resps_emb.embeddings.8.weight"],
	}

	mode_lvl_map = {
		'AR:0:0': 0,
		'NAR:0:1': 1,
		'NAR:1:2': 2,
		'NAR:2:3': 3,
		'NAR:3:4': 4,
		'NAR:4:5': 5,
		'NAR:5:6': 6,
		'NAR:6:7': 7,
		'NAR:0:0': 0,
		'len': 0,
	}

	embds = {}
	heads = {}
	n_embd = 1024

	with torch.no_grad():
		for k, v in io_map.items():
			start, end = v[0]
			classifier_idx = v[1]
			embd_name = v[2]
			
			if is_from_pretrained:
				n_vocab = end - start

				embds[k] = torch.nn.Embedding( n_vocab, n_embd ).to(model.embed_tokens.weight)
				embds[k].weight[:] = model.embed_tokens.weight[start:end, :]

				if classifier_idx >= 0:
					# NAR:0:0 does not have a masked token output
					if k == "resp|NAR:0:0":
						end -= 1
						n_vocab -= 1
					heads[k] = torch.nn.Linear( n_embd, n_vocab, bias=False ).to(hf_model.lm_head.weight)
					heads[k].weight[:] = hf_model.lm_head.weight[start:end, :]
			else:
				embd_weight = state_dict[embd_name].unsqueeze(0) if state_dict[embd_name].dim() == 1 else state_dict[embd_name]
				embds[k] = torch.nn.Embedding( embd_weight.shape[0], embd_weight.shape[1] ).to(device=device, dtype=dtype)
				embds[k].load_state_dict({ "weight": embd_weight })
				
				if classifier_idx >= 0:
					head_weight = state_dict[f'classifiers.proj.{classifier_idx}.weight']

					heads[k] = torch.nn.Linear( head_weight.shape[1], head_weight.shape[0], bias=False ).to(device=device, dtype=dtype)
					heads[k].load_state_dict({ "weight": head_weight })

	def create_inputs( phn, prom, lang=0, seq=None, mode="AR:0:0" ):
		rvq_l = mode_lvl_map[mode]

		inputs = torch.tensor([])
		pos_ids = torch.tensor([])
		attn_mask = torch.tensor([])

		seqs = []

		phn = torch.tensor(phn, device=device,dtype=torch.int32)
		prom = torch.tensor(prom, device=device,dtype=torch.int32)
		lang = torch.tensor([lang], device=device,dtype=torch.int32)
		rvq_l = torch.tensor([rvq_l], device=device,dtype=torch.int32)
		zero = torch.tensor([0], device=device,dtype=torch.int32)

		if mode == "len":
			seq = zero if not seq else torch.concat([zero, torch.tensor(seq, device=device, dtype=torch.int32)])
		elif seq:
			seq = torch.tensor(seq, device=device,dtype=torch.int32)
			seq = seq[:rvq_l, :] if rvq_l > 0 else seq

		sep_embd = embds["sep"](zero)
		phn_embd = embds["text"](phn)
		rvq_l_embd = embds["rvq_l"](rvq_l)
		lang_embd = embds["lang"](lang)
		prom_embd = torch.zeros(prom.shape[-1], n_embd, device=device, dtype=dtype)
		seq_embd = None

		for i, p in enumerate(prom):
			if i > rvq_l:
				break
			prom_embd += embds[f"prom|{i}"](p)

		if seq is not None:
			if mode == "len":
				seq_embd = embds["len"](seq)
			elif mode == "AR:0:0":
				seq_embd = embds["resp|AR:0:0"](seq)
			else:
				seq_embd = torch.zeros(seq.shape[-1], n_embd, device=device, dtype=dtype)
				for i, r in enumerate(seq):
					seq_embd += embds[f"resp|NAR:{i}:{i+1}"](r)

		seqs.append(torch.concat([phn_embd, sep_embd]))
		seqs.append(torch.concat([lang_embd, sep_embd]))
		seqs.append(torch.concat([rvq_l_embd, sep_embd]))
		seqs.append(torch.concat([prom_embd, sep_embd]))

		if seq_embd is not None:
			seqs.append(seq_embd)

		inputs = torch.concat(seqs)
		pos_ids = torch.tensor([ i for seq in seqs for i, _ in enumerate(seq) ], device=device, dtype=torch.int32)
		attn_mask = torch.tensor([ True for seq in seqs for i, _ in enumerate(seq) ], device=device, dtype=torch.bool)

		return inputs, pos_ids, attn_mask

	def generate( phn, prom, sequence=[], mode="resp|AR:0:0", max_tokens = 75 * 4, temperature = 1.0 ):
		lm_head = heads[mode]
		model._update_causal_mask = model._original_update_causal_mask

		n_outputs = 1
		stop_token = 1024
		if mode == "len":
			temperature = 0.0
			max_tokens = 5
			stop_token = 10
		elif mode != "resp|AR:0:0":
			temperature = 0.0
			max_tokens = len(sequence)+1
			n_outputs = len(sequence[0])

			model._update_causal_mask = model._update_noncausal_mask

		while len(sequence) < max_tokens:
			inputs, pos_ids, attn_mask = create_inputs( phn, prom, seq=sequence, mode=mode.split("|")[-1] )
			out = model(inputs_embeds=inputs.unsqueeze(0), position_ids=pos_ids.unsqueeze(0), attention_mask=attn_mask.unsqueeze(0))
			logits = lm_head(out[0]).float()

			logits = logits[0, -n_outputs:, :]
			t = Categorical(logits=logits / temperature).sample() if temperature > 0 else logits.argmax(dim=-1)
			if n_outputs > 1:
				sequence.append([ _.item() for _ in t ])
				break
			else:
				t = t[0]
				if stop_token in t:
					break
				sequence.append(t.item())
		return sequence

	# check embds
	if False:
		inputs, pos_ids, attn_mask = create_inputs( phn, prom, mode="len" )
		flattened = [ sum(embd).item() for embd in inputs ]

		for i, embd in enumerate( flattened ):
			print(f'{i}: ', pos_ids[i].item(), "\t", embd )


	# test len inferencing
	print( "len:", generate( phn, prom, mode="len" ) )

	# test ar ouptut
	if resp:
		resp = [ resp[0] ]
	else:
		resp = [ generate( phn, prom ) ]
		print( "AR:", resp )

	# test nar ouptut
	for i in range(1, 8):
		resp = generate( phn, prom, sequence=resp, mode=f"resp|NAR:{i-1}:{i}" )
		print( f"NAR:{i-1}:{i}: ", resp[-1] )

	decode_to_file( torch.tensor(resp, dtype=torch.int16, device=device).t(), "./data/test.wav" )