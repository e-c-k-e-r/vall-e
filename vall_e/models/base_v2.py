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

# aims to properly encode RVQ-encoded token sequence into an embedding
class ResidualAudioEncoder(nn.Module):
	def __init__(
		self,
		n_tokens: int,
		n_levels: int,
		token_dim: int,
		training: bool = True,
	):
		super().__init__()
		self.embs = nn.ModuleList([nn.Embedding(n_tokens, token_dim) for _ in range(n_levels)])
		self.pos_embedding = nn.Parameter(torch.randn(1, n_levels, token_dim)) # i still don't understand why this needs to be explicitly added instead of it being deduced in the embedding itself
		self.cross_attn = nn.MultiheadAttention( embed_dim=token_dim, num_heads=8, dropout=0.1 if training else 0.0, batch_first=True )
		self.proj = nn.Linear(token_dim, token_dim) # i don't understand why this is necessary

	def forward(self, xi: Tensor, dropout_mask = None, dropout_token = None ) -> Tensor:
		# empty
		if xi.shape[0] == 0:
			dim = self.embs[0].weight.shape[-1] # self.proj.weight.shape[0]
			return torch.zeros((0, dim), device=xi.device, dtype=xi.dtype)
		if dropout_mask is not None:
			xi = _dropout_codes( xi, dropout_mask, dropout_token )

		# ( seq_len, dim ) => ( seq_len, levels, dim )
		x = torch.stack([ emb(xi[:, i]) for i, emb in enumerate(self.embs) ], dim=1)
		x = x + self.pos_embedding
		attn, _ = self.cross_attn( x, x, x )
		x = x + attn
		x = self.proj( x.mean(dim=1) )

		return x
# aims to properly decode the last hidden states from a model into logits for an RVQ-encoded token sequence
class ResidualAudioDecoder(nn.Module):
	def __init__(
		self,
		d_model,
		vocab_size,
		resp_levels,
		training: bool = True,
		use_ln: bool = False,
	):
		super().__init__()

		self.projs = nn.ModuleList([nn.Sequential(
			(nn.LayerNorm(d_model) if use_ln else nn.Identity()),
			nn.Linear(d_model, d_model),
		) for _ in range(resp_levels)]) # per-level projs

		self.cross_attn = nn.MultiheadAttention( embed_dim=d_model, num_heads=8, dropout=0.1 if training else 0.0, batch_first=True ) # xattn so each level can attend to others per residual-ness
		self.head = nn.Linear(d_model, vocab_size) # final output head, i feel it would be better to have it per-level but i assume the proj handles it

	# forward for one sequence
	def _forward( self, x: Tensor ) -> Tensor:
		seq_len, resp_levels = x.shape[0], len(self.projs)
		x = torch.stack([proj(x) for proj in self.projs], dim=1)
		attn, _ = self.cross_attn( x, x, x )
		x = x + attn
		x = self.head( x )
		x = x.view( resp_levels, seq_len, -1 )
		return x

	# required to act on per sequence and not a batch due to headed-ness
	def forward( self, x_i: Tensor ) -> Tensor:
		return torch.stack([ self._forward(x) for x in x_i ], dim=0)

# the above, but for FSQ codecs, as each level is independent from one another
class FiniteAudioEncoder(nn.Module):
	def __init__(
		self,
		n_tokens: int,
		n_levels: int,
		token_dim: int,
		training: bool = True,
	):
		super().__init__()
		self.embs = nn.ModuleList([nn.Embedding(n_tokens, token_dim) for _ in range(n_levels)])
		self.pos_embedding = nn.Parameter(torch.randn(1, n_levels, token_dim))
		self.proj = nn.Linear(token_dim, token_dim)
		self.level_weights = nn.Parameter(torch.ones(n_levels))

	def forward(self, xi: Tensor, dropout_mask = None, dropout_token = None ) -> Tensor:
		# empty
		if xi.shape[0] == 0:
			dim = self.embs[0].weight.shape[-1] # self.proj.weight.shape[0]
			return torch.zeros((0, dim), device=xi.device, dtype=xi.dtype)
		if dropout_mask is not None:
			xi = _dropout_codes( xi, dropout_mask, dropout_token )

		x = torch.stack([ emb(xi[:, i]) for i, emb in enumerate(self.embs) ], dim=1)
		x = x + self.pos_embedding
		x = self.proj( x )
		
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
		d_ffn: int = 4,
		use_ln: bool = True,
		shared_levels: bool = False,
		training: bool = False,
	):
		super().__init__()
		self.n_levels = n_levels
		self.shared_levels = shared_levels

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

	def forward(self, x: Tensor) -> Tensor:
		batch_size, seq_len, _ = x.shape

		if not self.shared_levels:
			x = torch.stack([head(x) for head in self.head], dim=1)
		else:
			x = self.head(x)
			x = x.view(batch_size, seq_len, self.n_levels, -1)
			x = x.transpose(1, 2)

		return x

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
		predict_causally = config.experimental.predict_causally if config is not None else False
		monolithic_audio_encoder = config.experimental.monolithic_audio_encoder if config is not None else False
		audio_level_weights = [1.0 / (i + 1) for i in range(n_resp_levels)] # to-do: find the weights for FSQ
		logit_normalization = config.experimental.logit_normalization if config is not None else 0
		per_level_normalization = config.experimental.per_level_normalization if config is not None else True

		n_vocab = 256
		n_tasks = config.tasks if config is not None else 8
		n_langs = config.langs if config is not None else 2
		n_tones = config.tones if config is not None else 1

		if attention_backend == "auto":
			attention_backend = "sdpa"

		hf_attention = attention_backend
		HF_ATTENTIONS = ["eager", "sdpa", "flash_attention_2"]

		if attention_backend not in HF_ATTENTIONS:
			hf_attention = None
			if attention_backend not in AVAILABLE_ATTENTIONS:
				raise ValueError(f"Requesting attention `{attention_backend}` but is not available. Currently available: {AVAILABLE_ATTENTIONS}")

		# to-do: deduce nemo better-er
		if n_audio_tokens == 1000:
			# assume midrage contains important details
			center = n_resp_levels // 2
			audio_level_weights = [1.0 - abs(i - center) / n_resp_levels for i in range(n_resp_levels)]

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

		self.resp_parallel_training = resp_parallel_training
		self.predict_causally = predict_causally

		self.unified_position_ids = unified_position_ids
		self.inject_timestep_embedding = False # results in bad output
		self.masking_ratio = masking_ratio
		self.ignore_inputs_for_loss = ignore_inputs_for_loss
		self.noncausal_masks = noncausal_masks
		self.audio_level_weights = audio_level_weights
		self.logit_normalization = logit_normalization
		
		self.sep = nn.Parameter(torch.randn(d_model))

		self.phn_emb = ml.Embedding(n_phn_tokens, d_model)
		self.text_emb = ml.Embedding(n_text_tokens, d_model)
		self.langs_emb = ml.Embedding(n_langs, d_model) if n_langs > 0 else None
		self.tasks_emb = ml.Embedding(n_tasks, d_model) if n_tasks > 0 else None
		self.tones_emb = ml.Embedding(n_tones, d_model) if n_tones > 0 else None
		self.len_emb = ml.Embedding(11, d_model)

		self.audio_emb = None
		self.proms_emb = None
		self.resps_emb = None

		# to-do: deduce nemo-ness better-er
		if n_audio_tokens == 1000:
			AudioEncoder = FiniteAudioEncoder
			AudioDecoder = FiniteAudioDecoder
		else:
			AudioEncoder = ResidualAudioEncoder
			AudioDecoder = ResidualAudioDecoder

		if monolithic_audio_encoder:
			self.audio_emb = AudioEncoder(
				n_tokens=n_audio_tokens + 2, # stop + masked token
				n_levels=self.n_resp_levels,
				token_dim=d_model,
				training=training,
			)
		else:
			self.proms_emb = AudioEncoder(
				n_tokens=n_audio_tokens,
				n_levels=self.n_resp_levels,
				token_dim=d_model,
				training=training,
			)
			self.resps_emb = AudioEncoder(
				n_tokens=n_audio_tokens + 2, # stop + masked token
				n_levels=self.n_resp_levels,
				token_dim=d_model,
				training=training,
			)

		self.audio_decoder = AudioDecoder(
			d_model,
			(n_audio_tokens + 1),
			self.n_resp_levels,
			training=training,
			use_ln=per_level_normalization,
		)
		self.len_decoder = AuxDecoder( d_model, 11 )
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
				vocab_size=n_vocab,
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
				output_norm = not per_level_normalization, # moves the LN out to the decoder
				attn_mode = attention_backend,
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
				# throw an error so we don't silently train without this
				if self.len_emb is None:
					raise Exception(f"Requesting task `{task_type}` but corresponding embedding is not defined.")

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

				# insert output length tokens (if it exists)
				if len_list is not None and len_list[i] is not None:
					inputs[i].append( ( "len", len_list[i] ) )
				# "encode" length to tokens for 0-9 + stop
				elif resps_list is not None and resps_list[i] is not None:
					# yes this could be encoded better
					inputs[i].append( ( "len", torch.tensor([ 0 ] + [ int(i) for i in str( resps_list[i].shape[0]) ] + [ 10 ], device=device, dtype=torch.int16) ) )
				
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

			if self.audio_emb is not None:
				return self.audio_emb( input )
			
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
					if self.audio_emb is not None:
						embedding = self.audio_emb( input, dropout_mask=dropout_mask, dropout_token=self.mask_token )
					else:
						embedding = self.resps_emb( input, dropout_mask=dropout_mask, dropout_token=self.mask_token )
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
		level_weights = self.audio_level_weights

		# handles tasks where the prompt has task tokens injected in the middle
		def prompt_input_to_token( input, quant_level ):
			if isinstance(input, str):
				return torch.tensor( [ get_task_symmap()[input] ], device=device, dtype=torch.int16)

			return input

		def _calc_loss( logit, sequence, causal = True, level = None ):
			# filter tokens that exceed the vocab size
			sequence = torch.where( sequence >= logit.shape[-1], self.ignore_index, sequence )
			# drop if all tokens are ignored
			if torch.all(sequence == self.ignore_index):
				return None, None

			# shift if causal
			if causal or self.predict_causally:
				l = self.causal_size
				logit = logit[..., :-l, :] # shift the target so that token n...
				sequence = sequence[..., l:] # ...predicts token n + 1
			
			batched = sequence.dim() > 1

			# logit normalization
			if self.logit_normalization:
				# it would probably be better to unsqueeze then squeeze to avoid code duplication but who cares
				if not batched:
					logit = logit_normalization( logit, self.logit_normalization )
				else:
					for i, l in enumerate( logit ):
						logit[i] = logit_normalization( l, self.logit_normalization )

			# flatten batch
			if batched:
				logit = logit.reshape(-1, logit.shape[-1])
				sequence = sequence.reshape(-1)

			nll = None
			metrics = None

			if compute_hard_loss:
				reduction = 'mean' if not batched else 'none'
				weight = level_weights[level] if level is not None and not batched else 1

				nll = F.cross_entropy( logit, sequence, ignore_index=self.ignore_index, reduction=reduction ) * weight
				# manually weigh each level
				if batched:
					nll = nll.view( self.n_resp_levels, -1 ).mean(dim=-1) * torch.tensor(level_weights, device=device)

			if compute_acc:
				accuracy_metric = MulticlassAccuracy(
					logit.shape[-1],
					top_k = min(logit.shape[0], 10),
					average="micro",
					multidim_average="global",
					ignore_index = -100
				).to(logit.device)
				metrics = accuracy_metric( logit, sequence )
			
			return nll, metrics
		
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

					if logits[batch_index].dim() < 3 and token.dim() >= 2:
						token = token[..., 0]
				elif name == "resp":
					token = input

					# mask found, apply it
					if dropout_mask is not None:
						token = _dropout_codes( token, dropout_mask, self.ignore_index, swapped = True )
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
					token = torch.tensor( [ self.ignore_index ] * token.shape[0], device=device, dtype=torch.int16)

				# perform loss calculation on the individual piece
				if self.config.loss_factors:
					loss_factor = self.loss_factor(name)

					if loss_factor == 0.0:
						continue

					if logits[batch_index].dim() < 3:
						nll, metrics = _calc_loss( logits[batch_index][start:end], token.long(), causal )
					elif not self.resp_parallel_training:
						# cringe way to deduce "requested" level
						level = quant_level
						for i in range( self.n_resp_levels ):
							if classifier_level.endswith(f':{i}:{i}'):
								level = i
								break
						
						if name == "resp":
							name = f'{name}[{level}]'

						sequence = token if token.dim() <= 1 else token[:, level]
						nll, metrics = _calc_loss( logits[batch_index][level][start:end], sequence.long(), causal, level )
					else:
						sequence = token.t()
						nll, metrics = _calc_loss( logits[batch_index][:, start:end], sequence.long(), causal )

						if nll is not None:
							nll = nll.mean()

					loss_key = f'{name}.nll'
					acc_key = f'{name}.acc'
					if nll is not None:
						if loss_key not in loss:
							loss[loss_key] = []
						loss[loss_key].append( nll * loss_factor )
					
					if metrics is not None:
						if acc_key not in stats:
							stats[acc_key] = []
						stats[acc_key].append( metrics )
				# add to list
				else:
					target.append( token )
			

			# perform loss calculation on the entire sequence
			if not self.config.loss_factors:
				if logits[batch_index].dim() < 3:
					sequence = _join( target, torch.tensor(self.ignore_index, device=target[-1].device) )
					nll, metrics = _calc_loss( logits[batch_index], sequence, causal )
				elif not self.resp_parallel_training:
					# cringe way to deduce "requested" level
					level = 0
					for i in range( self.n_resp_levels ):
						if classifier_level.endswith(f':{i}:{i}'):
							level = i
							break

					sequence = [ x if x.dim() <= 1 else x[:, level] for x in target ] 
					sequence = _join( sequence, torch.tensor(self.ignore_index, device=sequence[-1].device) )
					nll, metrics = _calc_loss( logits[batch_index][level], sequence.long(), causal, level )
				else:
					nlls = []
					accs = []
					
					for level, logit in enumerate( logits[batch_index] ):
						sequence = [ x if x.dim() <= 1 else x[:, level] for x in target ] 
						sequence = _join( sequence, torch.tensor(self.ignore_index, device=sequence[-1].device) )
						nll, metrics = _calc_loss( logit, sequence, causal, level )

						if nll:
							nlls.append( nll )
						if metrics:
							accs.append( metrics )

					if nlls:
						nll = sum(nlls) / len(nlls)
					if accs:
						metrics = sum(accs) / len(accs)

				if nll is not None:
					if 'nll' not in loss:
						loss['nll'] = []
					loss["nll"].append( nll )
				
				if metrics is not None:
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
		
		m = mask.unsqueeze(dim=-1)

		# needs to be done here as we still have our raw inputs
		position_ids = self.inputs_to_position_ids( inputs, mask=mask ) if not self.unified_position_ids else None
		classifier_levels = self.get_input( inputs, name="classifier_level" )
		causal_levels = [ "len", "phn", "text" ] + [ f"AR:{_}:{_}" for _ in range( self.n_resp_levels) ]

		# right now limit to new versions because I need to retrain the model for noncausal masks...
		is_causal = [ l in causal_levels for l in classifier_levels ] if self.noncausal_masks else [ True for l in classifier_levels ]

		output = self._forward(
			inputs=x,
			mask=mask,
			state=state,
			is_causal=is_causal,
			position_ids=position_ids,
			output_attentions = output_attentions,
		)

		logits = [ logit for logit in output.logits ]
		hidden_states = output.hidden_states
		
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
		logits = [ hi[..., :li, :] for hi, li in zip(logits, map(len, x_list)) ]
		
		if not training:
			loss = None
			stats = None

			self.loss = None
			self.stats = None

		# compute loss if the target is given
		else:
			loss, stats = self.calc_loss( inputs=inputs, logits=logits, quant_levels=quant_levels )

			# include any additional losses (for example: MoE router)
			if output.loss is not None:
				loss["aux_loss"] = output.loss

			self.loss = loss
			self.stats = stats
			
		# rewrap, because we're modifying the logits here
		return Logits(logits, output.state, inputs, loss, output.attentions, hidden_states)

	def sample(
		self,
		logits: list[Tensor], # logit scores
		prev_list: list[Tensor] | None = None, # logit scores
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

		scores = None
		entropy = None

		if prev_list is not None:
			seq_lens = map(len, prev_list)
			logits = [ logit[-l:] for logit, l in zip(logits, seq_lens) ]
		# (AR chunkwise) return the last chunkwise piece
		elif self.causal:
			seq_lens = [ logit.shape[0] - self.causal_size for logit in logits ]
			logits = [ logit[-self.causal_size:] for logit in logits ]

		# argmax instead
		if temperature <= 0.0:
			res = [ logit.argmax(dim=-1) for logit in logits ]
		else:
			res = [ Categorical(logits=logit / temperature).sample() for logit in logits ]

		# calculate token probabilities
		scores = [
			[ F.softmax(logit[i, :], dim=-1)[token].item() for i, token in enumerate(tokens) ]
			for logit, tokens in zip(logits, res)
		]

		return Sampled(res, logits, scores, entropy)