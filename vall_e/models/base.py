import math
import torch
import torch.nn.functional as F
import traceback
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
from ..samplers import reptition_penalize, length_penalize, top_k_top_p_filtering, dynamic_temperature, top_k_logits_list, mirostat_sample

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

	# to-do: select quant level from given quant_levels tensor if given (i.e. through the resp_emb)
	# I imagine this is an oversight in the NAR.
	def forward(self, x_list: list[Tensor], quant_levels: Tensor | None = None) -> list[Tensor]:
		if len(x_list) == 0:
			return []

		# this "strategy" will reserve the weight[0] for te AR and weight[1:] for the NAR
		# the NAR cannot share RVQ-bin level 0 with the AR for the resp_emb
		if self.monolithic:
			w = self.weight[:1] if quant_levels is None else self.weight[1:]
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
class AudioEmbedding_Old(nn.Module):
	def __init__(
		self,
		l_tokens: int, # list of number of tokens (needed because AR resps includes stop token)
		token_dim: int, # dimensionality of the embedding
		levels: int | None = None, # number of RVQ-bins (I don't remember the specifics)
	):
		super().__init__()
		# array of embeddings
		#   proms are [0, prom_levels]
		#   resp are split to where [0] is for the AR, and [1:] are reserved for NAR
		self.embeddings = nn.ModuleList([nn.Embedding(n_tokens, token_dim) for n_tokens in l_tokens])
		# weight influencer for the influence for each level (desu this should be really useless because the weights in the embedding themselves should factor this)
		self.weight = nn.ParameterList([nn.Parameter( torch.Tensor([1]) ) for i in range(levels)]) if levels is not None else None

	def forward(self, xi: Tensor, quant_levels: Tensor | None = None ) -> Tensor:
		# prom
		if quant_levels is None and xi.shape[-1] > 1:
			x = sum( [ self.embeddings[k]( xi[:, k] ) * (self.weight[k] if self.weight is not None else 1) for k in range(xi.shape[-1]) ] )
		# AR resp
		elif quant_levels is None or quant_levels == 0:
			x = self.embeddings[0]( xi if len(xi.shape) == 1 else xi[:, 0] )
		# NAR resp
		else:
			x = sum( [ self.embeddings[k+1]( xi[:, k] ) * (self.weight[k+1] if self.weight is not None else 1) for k in range(xi.shape[-1]) ] )

		return x

class AudioEmbedding(nn.Module):
	def __init__(
		self,
		l_tokens: int, # list of number of tokens (needed because AR resps includes stop token)
		token_dim: int, # dimensionality of the embedding
		mode: str, # prom | resp
		sums: bool = True # whether to sum all previous layers of embeddings to factor in other RVQ bin levels (I do not know which way is better)
	):
		super().__init__()
		# array of embeddings
		#   proms are [0, prom_levels]
		#   resp are split to where [0] is for the AR, and [1:] are reserved for NAR
		self.embeddings = nn.ModuleList([nn.Embedding(n_tokens, token_dim) for n_tokens in l_tokens])
		# 
		self.mode = mode
		# 
		self.sums = sums

	# maintaining compat is hard
	def forward(self, xi: Tensor, quant_level: Tensor | None = None ) -> Tensor:
		if quant_level is None:
			quant_level = 0 if xi.dim() == 1 else xi.shape[-1] - 1

		# jank but needed
		if xi.dim() == 1:
			return self.embeddings[quant_level]( xi )
		
		offset = 0 if self.mode == "prom" else 1
		if self.sums and quant_level > 0:
			x = sum( [ self.embeddings[k + offset]( xi[:, k] ) for k in range( quant_level ) ] )
		else:
			k = quant_level - 1
			x = self.embeddings[k + offset]( xi if xi.dim() == 1 else xi[:, k] )

		return x

class Base(nn.Module):
	@property
	def causal(self) -> bool:
		raise NotImplementedError

	@property
	def arch_type(self) -> str:
		raise NotImplementedError

	@property
	def norm_type(self):
		raise NotImplementedError

	@property
	def n_prom_levels(self) -> int:
		raise NotImplementedError

	@property
	def n_resp_levels(self) -> int:
		raise NotImplementedError

	@property
	def n_max_levels(self) -> int:
		raise NotImplementedError
	
	@property
	def n_langs(self) -> int:
		raise NotImplementedError
	
	@property
	def n_tasks(self) -> int:
		raise NotImplementedError

	@property
	def n_tones(self) -> int:
		raise NotImplementedError

	@property
	def causal_size(self) -> int:
		raise NotImplementedError

	@property
	def interleave(self) -> bool:
		return False

	@property
	def monolithic(self) -> bool:
		return False

	@property
	def version(self) -> int:
		return 1

	@property
	def stop_token(self):
		if not self.causal:
			raise ValueError("Not using stop token!")
		return self.n_audio_tokens

	@property
	def ignore_index(self):
		return -100

	def loss_factor(self, k):
		if self.config is None:
			return 1.0
		return self.config.loss_factors[k] if k in self.config.loss_factors else 1.0

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
		self.gradient_checkpointing = self.config.gradient_checkpointing if self.config is not None else True

		self.n_text_tokens = n_text_tokens
		self.n_audio_tokens = n_audio_tokens

		self.d_model = d_model
		self.n_heads = n_heads
		self.n_layers = n_layers
		self.n_experts = n_experts
		
		self.l_padding = l_padding

		# +1 to include the stop token
		n_prom_tokens = n_audio_tokens
		n_resp_tokens = n_audio_tokens + self.causal_size

		self.text_emb = Embedding(n_text_tokens, d_model)
		self.langs_emb = None
		self.tones_emb = None
		self.tasks_emb = None
		self.rvq_level_emb = None

		if self.version == 1: # legacy
			n_prom_tokens += (self.n_tasks - 1) # old models have the task tokens in the prom
			self.proms_emb = MultiEmbedding(self.n_prom_levels, n_prom_tokens, d_model)
			self.resps_emb = MultiEmbedding(self.n_resp_levels, n_resp_tokens, d_model, monolithic=self.monolithic)
		elif self.version < 5:
			# [1024] * 8
			self.proms_emb = AudioEmbedding_Old(
				[n_prom_tokens] * self.n_prom_levels, d_model,
				levels=self.n_prom_levels if self.version > 3 else None,
			)
			# [1024 + STOP] + [1024] * 8
			self.resps_emb = AudioEmbedding_Old(
				[n_resp_tokens] + [n_resp_tokens - 1] * (self.n_resp_levels - 1), d_model,
				levels=self.n_resp_levels if self.version > 3 else None,
			)
		else:
			self.proms_emb = AudioEmbedding(
				[n_prom_tokens] * self.n_prom_levels, d_model,
				"prom",
				sums=self.config.audio_embedding_sums if self.config is not None else True
			)
			self.resps_emb = AudioEmbedding(
				[n_resp_tokens] + [n_resp_tokens - 1] * (self.n_resp_levels - 1), d_model,
				"resp",
				sums=self.config.audio_embedding_sums if self.config is not None else True
			)

		# useless since I actually removed using these with the input processing overhaul...
		if self.version >= 3:
			self.langs_emb = Embedding(self.n_langs, d_model) if self.n_langs > 0 else None
			self.tasks_emb = Embedding(self.n_tasks, d_model) if self.n_tasks > 0 else None
		# never actually got added... I kept forgetting to classify all my audio for speaker's tone
		if self.version >= 4:
			self.tones_emb = Embedding(self.n_tones, d_model) if self.n_tones > 0 else None

		# mamba requires this if a model does both AR and NAR tasks
		# this *might* help for AR and NAR tasks since we explicitly specify the current RVQ level for a sequence, rather than having it "encoded" in the embeddings
		# this ***might*** let me also unify the proms_emb and resps_embedding
		if self.version >= 5:
			self.rvq_level_emb = Embedding(self.n_resp_levels, d_model)

		# this would be nicer to be a stop token or live inside an embedding
		self.sep = nn.Parameter(torch.randn(d_model))

		# ick, there has to be a better way
		hf_attention = self.config.attention if self.config is not None else None

		if self.config.attention == "auto":
			if "flash" in AVAILABLE_ATTENTIONS:
				self.config.attention = "flash"
			elif "xformers" in AVAILABLE_ATTENTIONS:
				self.config.attention = "xformers"
			else:
				self.config.attention = "mem_efficient"

		if self.config.attention in ["xformers", "mem_efficient", "math", "flash"]:
			hf_attention = None
			if self.config.attention not in AVAILABLE_ATTENTIONS:
				raise ValueError(f"Requesting attention `{self.config.attention}` but is not available. Currently available: {AVAILABLE_ATTENTIONS}")


		if self.arch_type == "transformer":
			self.sin_emb = SinusoidalEmbedding(d_model)
			self.blocks = nn.ModuleList([TransformerBlock(
				d_model=d_model,
				n_heads=n_heads,
				p_dropout=p_dropout if training else 0.0,
				causal=self.causal,
				norm_type=self.norm_type,
				n_levels=self.n_resp_levels,
			) for _ in range(n_layers) ])
		elif self.arch_type in ["mistral", "mixtral"]:
			if n_experts <= 1:
				self.model = MistralModel(MistralConfig(
					vocab_size=n_resp_tokens,
					hidden_size=d_model,
					max_position_embeddings=75 * 60, # max-length of 60 seconds
					intermediate_size=d_model*4,
					num_hidden_layers=n_layers,
					num_attention_heads=n_heads,
					attention_dropout=p_dropout if training else 0.0,
					num_key_value_heads=self.config.kv_heads if self.config.kv_heads > 0 else n_heads,
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
					max_position_embeddings=75 * 60, # max-length of 60 seconds
					intermediate_size=d_model*4,
					num_hidden_layers=n_layers,
					num_attention_heads=n_heads,
					attention_dropout=p_dropout if training else 0.0,
					num_key_value_heads=self.config.kv_heads if self.config.kv_heads > 0 else n_heads,
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

			if self.gradient_checkpointing and not self.model.gradient_checkpointing:
				self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(
					use_reentrant=False
				))

			#if training:
			#	self.model.training = True
		elif self.arch_type == "llama":
			if n_experts <= 1:
				self.model = LlamaModel(LlamaConfig(
					vocab_size=n_resp_tokens,
					hidden_size=d_model,
					max_position_embeddings=75 * 60, # max-length of 60 seconds
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
			else:
				self.model = MixtralModel(MixtralConfig(
					vocab_size =n_resp_tokens,
					hidden_size=d_model,
					max_position_embeddings=75 * 60, # max-length of 60 seconds
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

			if self.gradient_checkpointing and not self.model.gradient_checkpointing:
				self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(
					use_reentrant=False
				))

			#if training:
			#	self.model.training = True
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
				n_layer=n_layers*2,
				d_intermediate=0,
				ssm_cfg={"layer": "Mamba2", "chunk_size":64} if self.arch_type == "mamba2" else {},
				rms_norm=True,
				fused_add_norm=True,
				residual_in_fp32=True,
				#attn_layer_idx=attn_layer_idx,
				#attn_cfg=attn_cfg,
				#initializer_cfg=initializer_cfg,
			)
			self.model.gradient_checkpointing = self.gradient_checkpointing
		else:
			raise RuntimeError(f'Unknown arch specified: {self.arch_type}')

		if self.config.attention in ["xformers", "auto", "mem_efficient", "math", "flash"]:
			self.model = ml.replace_attention( self.model, klass=LlamaAttention, target=LlamaAttention_Base, mode=self.config.attention )

		self.classifier = nn.Linear(d_model, n_resp_tokens)

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

	def _forward(
		self,
		inputs,
		mask = None,
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
		elif self.arch_type == "bitnet":
			x = self.model(x)

		# output projection layer with masking
		x = self.classifier(x) * mask

		return x, state, aux_loss

	def inputs(
		self,
		text_list: list[Tensor],
		proms_list: list[Tensor],
		resps_list: list[Tensor],

		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,

		quant_levels: Tensor | None = None
	):
		device = text_list[0].device
		batch_size = len(text_list)

		inputs = [ [] for _ in range(batch_size) ]
		for i in range(batch_size):
			quant_level = quant_levels[i] if quant_levels is not None else 0

			if text_list is not None:
				inputs[i].append( ( "text", text_list[i] ) )

			if self.rvq_level_emb is not None:
				inputs[i].append( ( "quant_level", torch.Tensor([ quant_level ]).to(device=device, dtype=torch.int16) ) )

			if proms_list is not None:
				inputs[i].append( ( "prom", proms_list[i] ) )
			if resps_list is not None:
				inputs[i].append( ( "resp", resps_list[i] ) )

		return inputs

	def inputs_to_embeddings(
		self,
		inputs: list,
		quant_levels: Tensor | None = None
	):
		x_list = []
		for batch_index, batch_input in enumerate(inputs):
			batch = []
			quant_level = quant_levels[batch_index] if quant_levels is not None else 0
			for name, input in batch_input:
				embedding = None
				if name == "text":
					embedding = self.text_emb( input )
				elif name == "quant_level" and self.rvq_level_emb is not None:
					embedding = self.rvq_level_emb( input )
				elif name == "lang" and self.langs_emb is not None:
					embedding = self.langs_emb( input )
				elif name == "prom":
					# get RVQ level 0, or up to targetted RVQ level inference
					embedding = self.proms_emb( input if input.dim() == 1 or quant_level == 0 else input[:, :quant_level], quant_level if self.version >= 5 else None )
				elif name == "tone" and self.tones_emb is not None:
					embedding = self.tones_emb( input )
				elif name == "resp":
					# get RVQ level 0, or up to targetted RVQ level inference
					embedding = self.resps_emb( input if input.dim() == 1 or quant_level == 0 else input[:, :quant_level], quant_level )
				else:
					continue

				batch.append(embedding)
	
			x_list.append( _join( batch, self.sep ) )

		return x_list

	def calc_loss(
		self,
		inputs: list,
		logits,
		
		quant_levels: Tensor | None = None,
	):
		# old, "naive" way, no loss factoring
		if not self.config.loss_factors:
			target_list = []

			for batch_index, batch in enumerate(inputs):
				quant_level = quant_levels[batch_index]
				prev_quant_level = 0 if quant_level == 0 else quant_level - 1
				target = []
				for name, input in batch:
					if name == "prom":
						# ignore prom, fill with mock tokens, because the prom embeddings don't directly map to tokens
						if self.version < 4 or (self.version >= 5 and self.config.audio_embedding_sums):
							target.append( torch.full_like(input[..., 0], self.ignore_index) )
						# we *CAN* directly map to proms
						else:
							target.append( input if input.dim() == 1 else input[:, prev_quant_level] )
					elif name == "resp":
						target.append( input if input.dim() == 1 else input[:, quant_level] )
					elif name in ["text", "quant_level", "lang", "tone"]:
						target.append( input )

				target_list.append( _join( target, torch.tensor(self.ignore_index, device=target[-1].device) ) )

			batch_size = len(target_list)
			# modify only for the AR so it can properly behave like a transformer
			for i in range(batch_size):
				if quant_levels is not None and quant_levels[i] > 0:
					continue

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
				self.stats = dict(
					acc = self.accuracy_metric( inputs, target ),
					# precision = self.precision_metric( inputs, target ),
				)
			else:
				self.loss = dict(
					nll = sum([ F.cross_entropy( inputs, targets, ignore_index=self.ignore_index ) for targets, inputs in zip( target_list, logits ) ]) / batch_size
				)
				self.stats = dict(
					acc = sum( [ self.accuracy_metric( inputs, targets ) for targets, inputs in zip( target_list, logits ) ] ) / batch_size
				)

			return

		"""
		# considerations:
		# * split losses does not maintain the entire sequence
		# * the first token is ignored for all pieces, rather than just the first text token (which is always provided)
		#     + the other way at least should keep it intact this way
		#     + extra logic might be required to instead offset from the end for the resp, rather than fit snuggly
		#     + this might just be a spook since the odds the very first token of the AR mattering is slim (although I swear I hear a very brief audio pop sometimes)
		"""
		self.loss = dict()
		self.stats = dict(acc = dict())

		info = {}
		batch_size = len( inputs )

		for i, batch in enumerate( inputs ):
			quant_level = quant_levels[i]
			prev_quant_level = 0 if quant_level == 0 else quant_level - 1

			it = 0
			for name, input in batch:
				# do not use resp
				if name == "resp":
					input = input if input.dim() == 1 else input[:, quant_level]
				# select prom level
				elif name == "prom":
					input = input[:, prev_quant_level]

				seq_len = input.shape[0]

				logit = logits[i][it:it+seq_len]
				it += seq_len + 1 # +1 to incorporate the separator
				
				# for the AR, shift sequence so that it predicts the next token
				#     (the NAR predicts the next token in place, so it's not necessary to do any modifications for it)
				if quant_level is None or quant_level == 0:
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
			if name not in ["text", "prom", "resp"]:
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
				self.stats["acc"][name] = sum( [ self.accuracy_metric( inputs, targets ) for targets, inputs in zip( batch["targets"], batch["logits"] ) ] ) / batch_size

	def forward(
		self,
		inputs: list,

		quant_levels: Tensor | None = None,
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


		x, state, aux_loss = self._forward(
			inputs=x,
			mask=m,
			state=state,
		)

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
		logits: list[Tensor],
		resps_list: list[Tensor],
		quant_levels: Tensor | None = None,

		temperature: float = 1.0,
		min_temperature: float = -1.0,
		top_k: int = -100,
		top_p: float = 1.0,

		repetition_penalty: float = 1.0,
		repetition_penalty_decay: float = 0.0,
		
		length_penalty: float = 0.0,
		
		beam_width: int = 0,

		mirostat: list[dict] | None = None,
	):
		if min_temperature < 0:
			min_temperature = temperature

		# (NAR) return the entire generated response
		# Parallel decoding relies on the last N tokens in the logits, because each token predicts the next RVQ layer in the same place (forgetfully obviously)
		if quant_levels is not None:
			logits = [ logit[-l:] for logit, l in zip(logits, map(len, resps_list)) ]
		# (AR chunkwise) return the last chunkwise piece
		elif self.causal:
			logits = [ logit[-self.causal_size:] for logit in logits ]

		devices = [ logit.device for logit in logits ]
		logits = [ logit.to(device="cpu", dtype=logit.dtype if logit.dtype != torch.float16 else torch.float32) for logit in logits ]

		# perform repetition penalizing	
		logits = [ reptition_penalize(logit, previous=resps[:, -1], factor=repetition_penalty, decay=repetition_penalty_decay) for logit, resps in zip( logits, resps_list ) ]

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