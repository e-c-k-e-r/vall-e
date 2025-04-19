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
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy, MulticlassPrecision

from .arch import *
from ..utils import ml, clamp, mean, logit_normalization
from ..samplers import *

# yuck, kind of needed
from ..data import get_task_symmap

import logging

_logger = logging.getLogger(__name__)

# these seem more elegant than a dict
Logits = namedtuple('Logits', ['logits', 'state', 'inputs', 'loss', 'attentions', 'hidden_states'])
Sampled = namedtuple('Sampled', ['ids', 'logits', 'scores', 'entropy'])
LossStats = namedtuple('LossStats', ['loss', 'stats'])

"""
from ..utils.pattern import DelayedPatternProvider, VALLEPattern
"""

summed_embeddings_task = [ "stt" ]
special_tasks = [ "len", "stt", "phn", "text" ]
non_tokened_names = ["task", "dropout_mask", "classifier_level"]
task_outputs = {
	"tts": "resp",
	"ns": "resp",
	"sr": "resp",
	"stt": "phn",
	"len": "len",
	"phn": "phn",
	"text": "text",
}

def _dropout_mask( input, p ):
	return (torch.rand(input.shape[0], device=input.device) < p)

def _create_mask(l, device):
	seq = torch.arange(max(l), device=device).unsqueeze(0)  # (1 t)
	stop = torch.tensor(l, device=device).unsqueeze(1)  # (b 1)
	return (seq < stop).float()  # (b t)

def _join(x: tuple[Tensor], sep: Tensor):
	ret = x[0]
	for i in range(1, len(x)):
		ret = torch.cat((ret, sep[None], x[i]), dim=0)
	return ret

def list_to_tensor(x_list: list[Tensor]):
	l = list(map(len, x_list))
	x = pad_sequence(x_list, batch_first=True)
	m = _create_mask(l, x_list[0].device)

	m = m.to(x).int()
	return x, m

def _dropout_codes( x, dropout_mask, dropout_token, swapped=False ):
	x = x.clone().detach()
	levels = x.shape[-1]
	for level in range( levels ):
		lhs = dropout_token if not swapped else x[..., level]
		rhs = x[..., level] if not swapped else dropout_token
		x[..., level] = torch.where( dropout_mask, lhs, rhs )
	return x

# aims to properly encode token sequences into an embedding
# despite being named for FSQ codecs, this works for RVQ codecs
class FiniteAudioEncoder(nn.Module):
	def __init__(
		self,
		n_tokens: int,
		n_levels: int,
		token_dim: int,
		monolithic: bool = False,
		use_ln: bool = True, # whether to perform a post-embedding pre-norm or not (I'm not sure if this is redundant)
		use_ffn: bool = True, # whether to employ a residual feed forward network or not
		use_level_weights: bool = False,

		d_model: int = None,
		d_ffn: int = 2, # feed forward expansion value
	):
		super().__init__()

		if not d_model:
			d_model = token_dim

		self.embs = nn.ModuleList([ml.Embedding(n_tokens, token_dim) for _ in range(n_levels)])
		
		# there needs to be some information when separating between the proms and the resps
		self.pos_embedding = nn.Parameter(torch.randn(2 if monolithic else 1, n_levels, token_dim) * 0.02)

		self.norm = nn.LayerNorm(token_dim) if use_ln else nn.Identity()

		if use_ffn:
			self.proj = nn.Sequential(
				nn.Linear(token_dim, token_dim * d_ffn),
				nn.GELU(),
				nn.Linear(token_dim * d_ffn, d_model),
			)
		elif token_dim != d_model:
			self.proj = nn.Linear(token_dim, d_model)
		else:
			self.proj = nn.Identity()

		self.level_weights = nn.Parameter(torch.ones(n_levels) / math.sqrt(n_levels)) if use_level_weights else None
		self.use_ffn = use_ffn

	def forward(self, xi: Tensor, dropout_mask = None, dropout_token = None, stability = None, mode = None ) -> Tensor:
		# empty
		if xi.shape[0] == 0:
			dim = self.embs[0].weight.shape[-1] # self.proj.weight.shape[0]
			return torch.zeros((0, dim), device=xi.device, dtype=xi.dtype)
		if dropout_mask is not None:
			xi = _dropout_codes( xi, dropout_mask, dropout_token )

		# some cronge
		if stability is None:
			stability = xi.dtype == torch.bfloat16

		x = torch.stack([ emb(xi[:, i]) for i, emb in enumerate(self.embs) ], dim=1)
		
		if mode == "prom":
			x = x + self.pos_embedding[0].unsqueeze(0)
		elif mode == "resp":
			x = x + self.pos_embedding[1].unsqueeze(0)
		else:
			x = x + self.pos_embedding

		x = self.norm(x)
		if self.use_ffn:
			x = x + self.proj( x )
		else:
			x = self.proj( x )

		if self.level_weights is None:
			x = x.sum(dim=1)
		else:
			weights = F.softmax(self.level_weights, dim=0).view(1, -1, 1)
			x = (x * weights).sum(dim=1)

		return x

# aims to decode the last hidden state into independent codebooks
# uses an MLP instead of Attn since it's not residual I guess (the problem with caving to consult claude-3-5-sonnet is that it will blindly agree with you if you suggest anything)
# optional per-level LN, might be beneficial
class FiniteAudioDecoder(nn.Module):
	def __init__(
		self,
		d_model: int,
		vocab_size: int,
		n_levels: int,

		d_ffn: int = 2, # feed forward expansion value
		use_ln: bool = True, # perform layer normalization here
		use_ffn: bool = True, # use a feed forward network post-norm pre-classifier
		shared_levels: bool = False, # whether to have one set of weights for all codebook levels, or separate weights for each layer
	):
		super().__init__()
		self.n_levels = n_levels
		self.shared_levels = shared_levels

		if use_ffn:
			if not shared_levels:
				self.head = nn.ModuleList([nn.Sequential(
					# ln
					(nn.LayerNorm(d_model) if use_ln else nn.Identity()),
					# ffn
					nn.Linear(d_model, d_ffn * d_model),
					nn.GELU(),
					nn.Linear(d_ffn * d_model, d_model),
					# head
					nn.Linear(d_model, vocab_size)
				) for _ in range(n_levels)])
			else:
				self.head = nn.Sequential(
					# ffn
					nn.Linear(d_model, d_ffn * d_model),
					nn.GELU(),
					nn.Linear(d_ffn * d_model, d_model),
					# head
					nn.Linear(d_model, vocab_size * n_levels)
				)
		else:
			if not shared_levels:
				self.head = nn.ModuleList([nn.Sequential(
					# ln
					(nn.LayerNorm(d_model) if use_ln else nn.Identity()),
					# head
					nn.Linear(d_model, vocab_size)
				) for _ in range(n_levels)])
			else:
				self.head = nn.Sequential(
					# head
					nn.Linear(d_model, vocab_size * n_levels)
				)

	def forward(self, x: Tensor) -> Tensor:
		batch_size, seq_len, _ = x.shape

		if not self.shared_levels:
			x = torch.stack([head(x) for head in self.head], dim=1)
		else:
			x = self.head(x)
			x = x.view(batch_size, seq_len, self.n_levels, -1)
			x = x.transpose(1, 2)

		return x

# the Residual variant doesn't seem to work well
# the Finite variant unironically works well for residual codecs
AudioEncoder = FiniteAudioEncoder
AudioDecoder = FiniteAudioDecoder

# handles simple output projections into logits for other tasks
class AuxDecoder(nn.Module):
	def __init__(
		self,
		d_model,
		vocab_size,
		name = None,
	):
		super().__init__()
		self.name = name
		self.head = nn.Linear( d_model, vocab_size )

	def forward(self, x: Tensor ) -> Tensor:
		x = self.head( x )
		return x

class Base_V2(nn.Module):
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

	def __init__(
		self,
		
		n_phn_tokens: int = 256,
		n_audio_tokens: int = 1024,
		n_text_tokens: int = 8575,

		d_model: int = 512,
		d_ffn: int = 4,
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

		if not attention:
			attention = config.attention if config is not None else "auto"

		n_resp_levels = config.resp_levels if config is not None else 8
		attention_backend = attention
		unified_position_ids = config.experimental.unified_position_ids if config is not None else True
		noncausal_masks = config.experimental.noncausal_masks if config is not None else False
		
		max_position_embeddings = config.experimental.max_position_embeddings if config is not None else (75 * 60 * 5)
		masking_ratio = config.experimental.masking_ratio if config is not None else False
		ignore_inputs_for_loss = config.experimental.ignore_inputs_for_loss if config is not None else False
		
		resp_parallel_training = config.experimental.resp_parallel_training if config is not None else True
		len_parallel_training = config.experimental.len_parallel_training if config is not None else False
		len_use_logits = config.experimental.len_use_logits if config is not None else True
		predict_causally = config.experimental.predict_causally if config is not None else False
		monolithic_audio_encoder = config.experimental.monolithic_audio_encoder if config is not None else False
		audio_level_loss_factors = config.experimental.audio_level_loss_factors if config is not None else "auto"
		len_loss_factor = config.experimental.len_loss_factor if config is not None else 0.00001
		logit_normalization = config.experimental.logit_normalization if config is not None else 0
		per_level_normalization = config.experimental.per_level_normalization if config is not None else True
		audio_decoder_ffn_expansion_size = config.experimental.audio_decoder_ffn_expansion_size if config is not None else 2
		audio_encoder_ffn_expansion_size = config.experimental.audio_encoder_ffn_expansion_size if config is not None else 2
		use_audio_encoder_ffn = config.experimental.use_audio_encoder_ffn if config is not None else True
		use_audio_encoder_norm = config.experimental.use_audio_encoder_norm if config is not None else True
		use_audio_encoder_level_weights = config.experimental.use_audio_encoder_level_weights if config is not None else True
		use_segmented_attention_mask = config.experimental.use_segmented_attention_mask if config is not None else True
		use_sliding_attention_mask = config.experimental.use_sliding_attention_mask if config is not None else True
		parallel_attention_mask_dropout = config.experimental.parallel_attention_mask_dropout if config is not None else 0.0

		n_vocab = 256
		n_tasks = config.tasks if config is not None else 8
		n_langs = config.langs if config is not None else 2
		n_tones = config.tones if config is not None else 1

		if audio_level_loss_factors == "auto":
			audio_level_loss_factors = "normal" if n_audio_tokens == 1000 else "decreasing"

		if audio_level_loss_factors == "decreasing":
			audio_level_loss_factors = [1.0 / (i + 1) for i in range(n_resp_levels)] 
		elif audio_level_loss_factors == "normal":
			if n_resp_levels == 8:
				audio_level_loss_factors = [0.5, 0.625, 0.75, 0.875, 0.875, 0.75, 0.625, 0.5]
			else:
				center = n_resp_levels // 2
				audio_level_loss_factors = [1.0 - abs(i - center) / n_resp_levels for i in range(n_resp_levels)]
			
			# to-do: proper cirriculum
			# prioritizes midrange, maybe good for epoch 0?
			# [0.5, 0.625, 0.75, 0.875, 0.875, 0.75, 0.625, 0.5]
			
			# deprioritizes midrange, good for epoch 1?
			# [0.875, 0.75, 0.625, 0.5, 0.5, 0.625, 0.75, 0.875]
		elif audio_level_loss_factors == "equal":
			audio_level_loss_factors = [1.0 for _ in range(n_resp_levels)]

		if attention_backend == "auto":
			attention_backend = "sdpa"

		hf_attention = attention_backend
		HF_ATTENTIONS = ["eager", "sdpa", "flash_attention_2"]

		if attention_backend not in HF_ATTENTIONS:
			hf_attention = None
			if attention_backend not in AVAILABLE_ATTENTIONS:
				raise ValueError(f"Requesting attention `{attention_backend}` but is not available. Currently available: {AVAILABLE_ATTENTIONS}")

		self.training = training
		self.teaching = False
		self.config = config

		self.n_phn_tokens = n_phn_tokens
		self.n_audio_tokens = n_audio_tokens
		self.n_text_tokens = n_text_tokens

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
		self.use_streamlined_calc_loss = True

		self.stop_token = self.n_audio_tokens
		self.mask_token = self.stop_token + 1

		self.causal = True
		self.version = self.config.version if self.config is not None else 5
		self.causal_size = self.config.experimental.causal_size if self.config is not None else (1 if self.causal else 0)

		self.arch_type = self.config.arch_type if self.config is not None else "llama"

		# check if requested arch is unavailable
		if self.arch_type in ERROR_ARCHES:
			raise ERROR_ARCHES[self.arch_type]

		# crunge
		if self.config is not None and config.teacher:
			self.teaching = True
			self.training = False

		self.predict_causally = predict_causally
		self.resp_parallel_training = resp_parallel_training
		self.len_parallel_training = len_parallel_training
		self.len_use_logits = len_use_logits
		self.unified_position_ids = unified_position_ids
		self.inject_timestep_embedding = False # results in bad output
		self.masking_ratio = masking_ratio
		self.ignore_inputs_for_loss = ignore_inputs_for_loss
		self.noncausal_masks = noncausal_masks
		self.audio_level_loss_factors = audio_level_loss_factors
		self.len_loss_factor = len_loss_factor
		self.logit_normalization = False # this actually kills the model's demasking capabilities
		self.use_segmented_attention_mask = use_segmented_attention_mask
		self.use_sliding_attention_mask = use_sliding_attention_mask
		self.parallel_attention_mask_dropout = parallel_attention_mask_dropout
		
		self.sep = nn.Parameter(torch.randn(d_model))

		self.phn_emb = ml.Embedding(n_phn_tokens, d_model)
		self.text_emb = ml.Embedding(n_text_tokens, d_model)
		self.langs_emb = ml.Embedding(n_langs, d_model) if n_langs > 0 else None
		self.tasks_emb = ml.Embedding(n_tasks, d_model) if n_tasks > 0 else None
		self.tones_emb = ml.Embedding(n_tones, d_model) if n_tones > 0 else None
		
		self.len_emb = nn.Parameter(torch.randn(d_model)) # ugh

		self.audio_emb = None
		self.proms_emb = None
		self.resps_emb = None

		if monolithic_audio_encoder:
			self.audio_emb = AudioEncoder(
				n_tokens=n_audio_tokens + 2, # stop + masked token
				n_levels=self.n_resp_levels,
				token_dim=d_model,
				monolithic=True,
				d_ffn=audio_encoder_ffn_expansion_size,
				use_ln=use_audio_encoder_norm,
				use_ffn=use_audio_encoder_ffn,
				use_level_weights=use_audio_encoder_level_weights,
			)

			self.proms_emb = lambda *args, **kwargs: self.audio_emb( *args, **kwargs, mode="prom" )
			self.resps_emb = lambda *args, **kwargs: self.audio_emb( *args, **kwargs, mode="resp" )
		else:
			self.proms_emb = AudioEncoder(
				n_tokens=n_audio_tokens,
				n_levels=self.n_resp_levels,
				token_dim=d_model,
				d_ffn=audio_encoder_ffn_expansion_size,
				use_ln=use_audio_encoder_norm,
				use_ffn=use_audio_encoder_ffn,
				use_level_weights=use_audio_encoder_level_weights,
			)
			self.resps_emb = AudioEncoder(
				n_tokens=n_audio_tokens + 2, # stop + masked token
				n_levels=self.n_resp_levels,
				token_dim=d_model,
				d_ffn=audio_encoder_ffn_expansion_size,
				use_ln=use_audio_encoder_norm,
				use_ffn=use_audio_encoder_ffn,
				use_level_weights=use_audio_encoder_level_weights,
			)

		self.audio_decoder = AudioDecoder(
			d_model,
			(n_audio_tokens + 1),
			self.n_resp_levels,
			use_ln=per_level_normalization,
			d_ffn=audio_decoder_ffn_expansion_size,
		)
		self.len_decoder = AuxDecoder( d_model, 1 if not len_use_logits else (10 * 5) )
		self.phn_decoder = AuxDecoder( d_model, n_phn_tokens )
		self.text_decoder = AuxDecoder( d_model, n_text_tokens )

		# override any requested padding size
		if attention_backend == "flash_attn_v100":
			self.l_padding = 32
		elif attention_backend == "fused_attn":
			self.l_padding = 128

		if self.arch_type in ["none"]:
			self.model = None
		elif self.arch_type in ["llama"]:
			self.model_config = LlamaConfig(
				vocab_size=0, # n_vocab,
				hidden_size=d_model,
				max_position_embeddings=max_position_embeddings,
				intermediate_size=d_model*d_ffn,
				num_hidden_layers=n_layers,
				num_attention_heads=n_heads,
				attention_dropout=p_dropout if training else 0.0,
				num_key_value_heads=n_heads,
				hidden_act="gelu",
				is_encoder_decoder=False,
				is_decoder=True,
				#gradient_checkpointing=self.gradient_checkpointing,

				# extra parameters
				output_norm = not per_level_normalization, # moves the LN out to the decoder
				attn_mode = attention_backend,
				causal = self.causal,
			)
			self.model = LlamaModel(self.model_config)

			if self.gradient_checkpointing and not self.model.gradient_checkpointing:
				self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(
					use_reentrant=False
				))
		else:
			raise RuntimeError(f'Unknown arch specified: {self.arch_type}')

		if hasattr( self.model, "embeddings" ):
			del self.model.embeddings

	def _forward(
		self,
		inputs,
		mask = None,
		is_causal = None,
		position_ids = None,
		
		state = None,
		
		output_attentions = False,
		output_hidden_states = False,
	):
		x = inputs
		m = mask #.squeeze(-1).int()
		
		aux_loss = None
		attentions = None
		hidden_states = None

		if self.arch_type in ["none"] or self.model is None:
			...
		# HF transformer derived model
		elif self.arch_type in ["llama"]:
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
				router_logits = output["router_logits"]
				aux_loss = self.model.config.router_aux_loss_coef * load_balancing_loss_func( router_logits, self.model.config.num_local_experts, self.model.config.num_experts_per_tok, m )
		else:
			hidden_states = self.model(x)

		# process it into a format that I like
		if output_hidden_states:
			# hidden_states is actually layers + 1, as hidden_states[0] == embedding...........
			hidden_states = [ state for state in hidden_states[1:] ]
			# apply normalization to these states (to-do: check if this matters)
			# but skip the last state, as it already is normalized
			hidden_states = [ x if i == self.n_layers - 1 else self.model.norm(output.hidden_states[i]) for i, state in enumerate( hidden_states ) ]

		return Logits(x, state, inputs, aux_loss, attentions, hidden_states)

	# takes a bunch of separate lists and parses them into an ordered array of tuples to guide input sequence creation
	def inputs(
		self,
		phns_list: list[Tensor] | None = None,
		text_list: list[Tensor] | None = None,

		proms_list: list[Tensor] | None = None,
		resps_list: list[Tensor] | None = None,

		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
		len_list: list[Tensor] | None = None,
		task_list: list[str] | None = None,
		time_list: list[Tensor] | None = None,

		quant_levels: int | list[int] | Tensor | None = None
	):
		if phns_list and phns_list[0] is not None:
			device = phns_list[0].device
			batch_size = len(phns_list)
		elif text_list and text_list[0] is not None:
			device = text_list[0].device
			batch_size = len(text_list)
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
			# Sequence: <phn><sep><rvq lvl><sep><prom><sep><resp>
			# prom /may/ include <task> tokens inside to help guide things, per SpeechX
			if task_type in get_task_symmap() and task_type not in special_tasks:
				# insert the phn prompt
				if phns_list is not None and phns_list[i] is not None:
					inputs[i].append( ( "phn", phns_list[i] ) )
				elif text_list is not None and text_list[i] is not None:
					inputs[i].append( ( "text", text_list[i] ) )
				# insert lang token if we're trained for it
				if lang_list is not None and lang_list[i] is not None:
					inputs[i].append( ( "lang", lang_list[i] ) )
				# insert input audio prompt
				if proms_list is not None and proms_list[i] is not None:
					inputs[i].append( ( "prom", proms_list[i] ) )
				# insert tone token if we're trained for it
				if tone_list is not None and tone_list[i] is not None:
					inputs[i].append( ( "tone", tone_list[i] ) )
				# insert timestep token
				if timestep is not None:
					p = self.masking_ratio

					# store dropout mask (if training, as this gets used later to mask the input embeddings if provided)
					if self.training and p > 0:
						dropout_mask = _dropout_mask( resps_list[i], p )
						inputs[i].append( ("dropout_mask", dropout_mask ) )
				# insert the current output response
				if resps_list is not None and resps_list[i] is not None:
					inputs[i].append( ( "resp", resps_list[i] ) )
				
				classifier_level = f"{'N' if timestep is not None else ''}AR:{quant_level}:{quant_level}"

				inputs[i].append( ("classifier_level", classifier_level) )
			# Audio length prediction task
			# Sequence: <phn><sep><rvq lvl><prom><sep><len>
			elif task_type == "len":
				# insert the phn prompt
				if phns_list is not None and phns_list[i] is not None:
					inputs[i].append( ( "phn", phns_list[i] ) )
				elif text_list is not None and text_list[i] is not None:
					inputs[i].append( ( "text", text_list[i] ) )
				# insert lang token if we're trained for it
				if lang_list is not None and lang_list[i] is not None:
					inputs[i].append( ( "lang", lang_list[i] ) )
				# insert input audio prompt
				if proms_list is not None and proms_list[i] is not None:
					inputs[i].append( ( "prom", proms_list[i] ) )
				# insert tone token if we're trained for it
				if "tone" in self.capabilities and tone_list is not None and tone_list[i] is not None:
					inputs[i].append( ( "tone", tone_list[i] ) )
				# insert len marker
				if resps_list is not None:
					inputs[i].append( ( "len", torch.tensor([resps_list[i].shape[0]]) ) )
				else:
					inputs[i].append( ( "len", torch.tensor([0]) ) )
				
				inputs[i].append( ("classifier_level", "len") )
			# Speech-to-Text prediction task
			# Sequence: <resp><sep><rvq lvl><sep><phn>
			elif task_type == "stt":
				# insert the input response
				if resps_list is not None and resps_list[i] is not None:
					inputs[i].append( ( "resp", resps_list[i] ) )
				# insert lang token if we're trained for it
				if lang_list is not None and lang_list[i] is not None:
					inputs[i].append( ( "lang", lang_list[i] ) )
				# insert the output phn prompt
				if phns_list is not None and phns_list[i] is not None:
					inputs[i].append( ( "phn", phns_list[i] ) )

				inputs[i].append( ("classifier_level", "phn") )
			# Text phonemizing task
			# Sequence: <text><sep><lang><sep><phonemes>
			elif task_type == "phn":
				# insert the phn prompt
				if text_list is not None and text_list[i] is not None:
					inputs[i].append( ( "text", text_list[i] ) )
				# insert lang token if we're trained for it
				if lang_list is not None and lang_list[i] is not None:
					inputs[i].append( ( "lang", lang_list[i] ) )
				# insert the phn prompt
				if phns_list is not None and phns_list[i] is not None:
					inputs[i].append( ( "phn", phns_list[i] ) )

				inputs[i].append( ("classifier_level", "phn") )
			# Text de-phonemizing task
			# Sequence: <text><sep><lang><sep><phonemes>
			elif task_type == "text":
				# insert the phn prompt
				if phns_list is not None and phns_list[i] is not None:
					inputs[i].append( ( "phn", phns_list[i] ) )
				# insert lang token if we're trained for it
				if lang_list is not None and lang_list[i] is not None:
					inputs[i].append( ( "lang", lang_list[i] ) )
				# insert the phn prompt
				if text_list is not None and text_list[i] is not None:
					inputs[i].append( ( "text", text_list[i] ) )
				
				inputs[i].append( ("classifier_level", "text") )
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
				return self.tasks_emb( torch.tensor( [ get_task_symmap()[input] ], device=device, dtype=torch.int16) )

			return self.proms_emb( input )

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
				elif name == "phn":
					embedding = self.phn_emb( input )

					device = embedding.device
				elif name == "text" and self.text_emb is not None:
					embedding = self.text_emb( input )

					device = embedding.device
				elif name == "quant_level" and self.rvq_l_emb is not None:
					embedding = self.rvq_l_emb( input )
				elif name == "lang" and self.langs_emb is not None:
					embedding = self.langs_emb( input )
				elif name == "prom":
					proms = [ input ] if isinstance(input, torch.Tensor) else input
					embedding = torch.cat( [ prompt_input_to_embedding( input, quant_level ) for input in proms if input is not None ] )
				elif name == "tone" and self.tones_emb is not None:
					embedding = self.tones_emb( input )
				elif name == "resp":
					embedding = self.resps_emb( input, dropout_mask=dropout_mask, dropout_token=self.mask_token )
				elif name == "timestep" and self.time_emb is not None:
					embedding = self.time_emb( input )
				elif name == "len" and self.len_emb is not None:
					# singleton marker
					embedding = self.len_emb[None]
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
		logits_aux = None,
		
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

			return input

		# handles "tokenizing" an integer
		def tokenize_duration( seq_lens, device, dtype=torch.int64 ):
			return torch.tensor( [ [ int(i) for i in str( l ).zfill(5) ] for l in seq_lens], device=device, dtype=dtype)

		k_lo, k_hi = 1, 20
		level_loss_factors = self.audio_level_loss_factors
		
		# this could be one array of tuples but can't be assed
		loss_targets = []
		loss_logits = []
		loss_factors = []
		loss_names = []

		resp_durations = []

		for batch_index, batch in enumerate(inputs):
			quant_level = quant_levels[batch_index]
			causal = False
			task_type = "tts"
			dropout_mask = None
			classifier_level = None

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
					parts = [ prompt_input_to_token( i, quant_level ) for i in proms if i is not None ]
					for i, p in enumerate( parts ):
						if p.dim() == 1:
							parts[i] = p.repeat(p.shape[0], self.n_resp_levels)
					token = torch.cat( parts )

					if logits[batch_index].dim() < 3 and token.dim() >= 2:
						token = token[..., 0]
				elif name == "resp":
					token = input

					# mask found, apply it
					if dropout_mask is not None:
						token = _dropout_codes( token, dropout_mask, self.ignore_index, swapped = True )
				elif name == "len":
					size = input[0].item()
					token = torch.tensor([ int(i) for i in str( size ).zfill(5) ], device=device, dtype=torch.int64)
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
					continue

				sequence = token
				if token.dim() == 1:
					loss_factor = self.loss_factor(name)
					if loss_factor == 0.0:
						continue
					
					logit = logits[batch_index][start:end]

					"""
					if self.logit_normalization:
						logit = logit_normalization( logit, self.logit_normalization )
					"""

					if causal or self.predict_causally:
						l = self.causal_size
						loss_targets.append( token[l:].long() ) # shift the target so that token n...
						loss_logits.append( logit[..., :-l, :] ) # ...predicts token n + 1
					elif name == "len":
						loss_targets.append( token.long() )
						loss_logits.append( logit.squeeze(0) )
					else:
						loss_targets.append( token.long() )
						loss_logits.append( logit )
					
					loss_factors.append( loss_factor )
					loss_names.append( name )
				else:
					if name == "resp":
						resp_durations.append( token.shape[0] )
					for level in range( self.n_resp_levels ):
						if not self.resp_parallel_training and not classifier_level.endswith(f':{level}:{level}'):
							continue

						logit = logits[batch_index][level][start:end]

						"""
						if self.logit_normalization:
							logit = logit_normalization( logit, self.logit_normalization )
						"""

						if causal or self.predict_causally:
							l = self.causal_size
							loss_targets.append( token[l:, level].long() ) # shift the target so that token n...
							loss_logits.append( logit[..., :-l, :] ) # ...predicts token n + 1
						else:
							loss_targets.append( token[:, level].long() )
							loss_logits.append( logit )


						loss_factors.append( level_loss_factors[level] )
						loss_names.append( name )

				break

		# fill in gap to make dim=-1 equal by filling in with -infs
		# the old implementation inherently does this through the master Classifiers class
		# but it needs to explicitly be done here
		dim_neg_1 = 0
		for batch_index, logit in enumerate( loss_logits ):
			dim_neg_1 = max( dim_neg_1, logit.shape[-1] )
		
		for batch_index, logit in enumerate( loss_logits ):
			if dim_neg_1 == logit.shape[-1]:
				continue

			loss_logits[batch_index] = torch.cat([logit, torch.full( (logit.shape[0], dim_neg_1 - logit.shape[-1]), -float("inf"), device=logit.device, dtype=logit.dtype) ], dim=-1 )


		loss_target = torch.cat( loss_targets )
		loss_logit = torch.cat( loss_logits )

		nll = None
		acc_k_lo = None
		acc_k_hi = None

		if compute_hard_loss:
			nll = 0
			nlls = F.cross_entropy( loss_logit, loss_target, reduction='none', ignore_index=self.ignore_index )

			# not my best code
			it = 0
			weights = 0
			bsz = len( loss_targets )
			for seq, loss_factor in zip( loss_targets, loss_factors ):
				seq_len = seq.shape[0]
				start = it
				it += seq_len
				end = it

				nll += nlls[start:end].mean() * loss_factor
				weights += loss_factor

			# normalize by batch
			nll /= bsz
			# re-scale by summed weights
			nll /= (weights / bsz)
			# no this isn't redundant I swear, it'll propagate properly

		if compute_acc:
			n_vocab = loss_logit.shape[-1]
			if n_vocab >= k_lo:
				accuracy_metric = MulticlassAccuracy(
					n_vocab,
					top_k = 1,
					average="micro",
					multidim_average="global",
					ignore_index = -100
				).to(loss_logit.device)
				acc_k_lo = accuracy_metric( loss_logit, loss_target )

			if n_vocab >= k_hi:
				accuracy_metric = MulticlassAccuracy(
					n_vocab,
					top_k = 20,
					average="micro",
					multidim_average="global",
					ignore_index = -100
				).to(loss_logit.device)
				acc_k_hi = accuracy_metric( loss_logit, loss_target )

		# to-do: re-add reporting split losses
		if nll is not None:
			if 'nll' not in loss:
				loss['nll'] = []
			loss["nll"] = nll
		
		if acc_k_lo is not None:
			acc_k_lo = acc_k_lo.mean()
			if f'acc[k={k_lo}]' not in stats:
				stats[f'acc[k={k_lo}]'] = []
			stats[f"acc[k={k_lo}]"] = acc_k_lo

		if acc_k_hi is not None:
			acc_k_hi = acc_k_hi.mean()
			if f'acc[k={k_hi}]' not in stats:
				stats[f'acc[k={k_hi}]'] = []
			stats[f"acc[k={k_hi}]"] = acc_k_hi

		# check if len logits are provided
		if logits_aux is not None:
			len_factor = self.len_loss_factor # 0.001 # to-do: user adjustable (it's really small because mse_loss causes wildly bigly losses)
			aux_loss_logit = torch.cat( logits_aux )

			if self.len_use_logits:
				aux_loss_target = torch.tensor( [ [ int(i) for i in str( l ).zfill(5) ] for l in resp_durations ], device=aux_loss_logit.device, dtype=torch.int64).flatten()
				loss['len'] = F.cross_entropy( aux_loss_logit, aux_loss_target ) * len_factor
			else:
				aux_loss_target = torch.tensor( resp_durations, device=aux_loss_logit.device, dtype=aux_loss_logit.dtype ) / self.audio_frames_per_second
				loss['len'] = F.mse_loss( aux_loss_logit, aux_loss_target ) * len_factor

		return LossStats(loss, stats)

	def forward(
		self,
		inputs: list,

		quant_levels: list[int] | None = None,
		state: dict | list | None = None,
		
		output_attentions: bool = False,
		output_hidden_states: bool = False,
	):	
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
		
		# needs to be done here as we still have our raw inputs
		position_ids = self.inputs_to_position_ids( inputs, mask=mask ) if not self.unified_position_ids else None
		classifier_levels = self.get_input( inputs, name="classifier_level" )
		causal_levels = [ "phn", "text" ] + [ f"AR:{_}:{_}" for _ in range( self.n_resp_levels) ]

		# right now limit to new versions because I need to retrain the model for noncausal masks...
		is_causal = [ l in causal_levels for l in classifier_levels ] if self.noncausal_masks else [ True for l in classifier_levels ]

		if self.parallel_attention_mask_dropout > 0:
			is_causal = [ True if random.random() < self.parallel_attention_mask_dropout else m for m in is_causal ]

		# create special masks
		# to-do, create it if mixed (although I expect this model to be purely non-causal)

		text_window = 32 if self.use_sliding_attention_mask else 0
		audio_window = self.audio_frames_per_second // 2 if self.use_sliding_attention_mask else 0

		aux_lens = []
		aux_windows = []
		# fill aux lens
		for batch_index, batch_input in enumerate( inputs ):
			lens = [2, 0, 0]
			windows = [text_window, audio_window, audio_window]

			for name, input in batch_input:
				if isinstance(input, list):
					shape = sum( [ i.shape[0] for i in input if isinstance(i, torch.Tensor) ] )
				elif not isinstance(input, torch.Tensor):
					continue
				else:
					shape = input.shape[0]

				if name in ["phn", "text"]:
					lens[0] = shape + 1
				elif name == "lang":
					lens[0] += 2
				elif name == "prom":
					lens[1] = shape + 1
				elif name == "tone":
					lens[1] += 2
				elif name == "len":
					lens[2] = 2
				elif name == "resp":
					lens[2] = shape

			aux_lens.append( lens )
			aux_windows.append( windows )

		if self.use_segmented_attention_mask and not any(is_causal):
			mask = self.model._update_segmented_mask( mask, x, aux_lens, window_sizes=aux_windows )

		output = self._forward(
			inputs=x,
			mask=mask,
			state=state,
			is_causal=is_causal,
			position_ids=position_ids,
			output_attentions = output_attentions,
		)

		hidden_states = output.hidden_states
		
		# logits = self.audio_decoder( output.logits )

		logits = [ logit for logit in output.logits ]
		logits_aux = None

		grouped_logits = {}
		
		for batch_index in range( batch_size ):
			classifier_level = classifier_levels[batch_index]
			if classifier_level.startswith("AR:") or classifier_level.startswith("NAR:"):
				classifier_level = "audio"

			if classifier_level not in ["audio", "phn", "text", "len"]:
				continue
			
			if classifier_level not in grouped_logits:
				grouped_logits[classifier_level] = []
			
			grouped_logits[classifier_level].append(batch_index)

		for classifier_level, decoders_indices in grouped_logits.items():
			if classifier_level == "audio":
				head = self.audio_decoder
			elif classifier_level == "phn":
				head = self.phn_decoder
			elif classifier_level == "text":
				head = self.text_decoder
			elif classifier_level == "len":
				head = self.len_decoder

			decoders_logits = torch.stack([ logits[batch_index] for batch_index in decoders_indices ])
			decoders_logits = head( decoders_logits )
			for batch_index, logit in zip( decoders_indices, decoders_logits ):
				logits[batch_index] = logit

		# Remove padding
		logits = [ logit[..., :l, :] for logit, l in zip(logits, map(len, x_list)) ]

		for batch_index, classifier_level in enumerate( classifier_levels ):
			if classifier_level != "len":
				continue

			logits[batch_index] = logits[batch_index].view(-1, 5, 10)

		if not training:
			loss = None
			stats = None

			self.loss = None
			self.stats = None

			# this can all technically be grabbed outside of this forward and manually invoke len_decoder on the last hidden states
			tasks = self.get_input( inputs, name="task" )

			# grab duration if no resp is provided or len task is requested
			if tasks[0] == "len":
				# do duration prediction
				logits_aux = self.len_decoder( output.logits )
				# it's more accurate this way
				logits_aux = [ logit[..., -1, :] for logit, aux_len in zip(logits_aux, aux_lens) ]
				# reshape logits
				if self.len_use_logits:
					# get tokens
					logits_aux = [ logit.view(5, 10).argmax(dim=-1) for logit in logits_aux ]
					# stitch
					logits_aux = [ int("".join([ str(t.item()) for t in logit ])) / self.audio_frames_per_second for logit in logits_aux ]

				logits = logits_aux
		# compute loss if the target is given
		else:
			# do duration prediction
			if self.len_parallel_training:
				logits_aux = self.len_decoder( output.logits )
				# only keep the input
				logits_aux = [ logit[..., aux_len[0] + aux_len[1], :] for logit, aux_len in zip(logits_aux, aux_lens) ]

				# reshape logits
				if self.len_use_logits:
					logits_aux = [ logit.view(5, 10) for logit in logits_aux ]
			else:
				logits_aux = None

			loss, stats = self.calc_loss( inputs=inputs, logits=logits, logits_aux=logits_aux, quant_levels=quant_levels )

			# include any additional losses (for example: MoE router)
			if output.loss is not None:
				loss["aux_loss"] = output.loss

			self.loss = loss
			self.stats = stats
			
		# rewrap, because we're modifying the logits here
		return Logits(logits, output.state, inputs, loss, output.attentions, hidden_states)

	def sample(
		self,
		logits: Tensor, # logit scores
		prev_list: Tensor | None = None,
		len_list: Tensor | None = None,
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
		device = logits[0].device

		if min_temperature < 0:
			min_temperature = temperature

		scores = None
		entropy = None
		causal = False

		if prev_list is not None:
			seq_lens = [ prev.shape[0] for prev in prev_list ]
		elif len_list is not None:
			seq_lens = len_list
		elif self.causal:
			seq_lens = [ self.causal_size for _ in range( batch_size) ]
			causal = True

		logits = [ logit[..., -l:, :] for l, logit in zip(seq_lens, logits) ]

		# perform min_p filtering of our logits
		if min_p > 0.0:
			logits = [ min_p_filtering(logit, min_p=min_p) for logit in logits ]

		# perform top_k/top_p filtering of our logits
		if top_k > 0 or top_p < 1.0:
			logits = [ top_k_top_p_filtering(logit, top_k=top_k, top_p=top_p) for logit in logits ]	

		# do top-no logit processing
		if top_no > 0.0:
			logits = [ top_no_logits_processing(logit) for logit in logits ]

		probabilities = [ F.softmax(logit, dim=-1) for logit in logits ]
		scores = [ torch.max(prob, -1)[0] for prob in probabilities ]

		if temperature <= 0.0:
			res = [ prob.argmax(dim=-1) for prob in probabilities]
		else:
			res = [ Categorical(logits=logit / temperature).sample() for logit in logits ]

		# we only need the scores for NAR demasking, but AR breaks and I cannot be assed to handle it right now
		scores = [
			torch.gather(prob, 2, tokens.unsqueeze(-1)).squeeze(-1)
			for prob, tokens in zip(probabilities, res)
		] if not causal else []

		return Sampled(res, logits, scores, entropy)