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

from ..utils import wrapper as ml

from ..samplers import reptition_penalize, length_penalize, top_k_top_p_filtering, dynamic_temperature, top_k_logits_list, mirostat_sample

try:
	from .transformer import SinusoidalEmbedding, Block as TransformerBlock
except Exception as e:
	print("Error importing `transformer` arch:", e)
	pass

try:
	#from .retnet import RetNetDecoder, RetNetConfig
	from .retnet_ts import RetNetDecoder, RetNetConfig
except Exception as e:
	print("Error importing `retnet` arch:", e)
	pass

from .retnet_hf import RetNetDecoder as RetNetDecoder_HF, RetNetConfig as RetNetConfig_HF
"""
try:
except Exception as e:
	print("Error importing `retnet-hf` arch:", e)
	pass
"""

try:
	from transformers import LlamaModel, LlamaConfig
except Exception as e:
	print("Error importing `llama` arch:", e)
	pass

try:
	from transformers import MistralModel, MistralConfig
except Exception as e:
	print("Error importing `mistral` arch:", e)
	pass

try:
	from bitnet.bit_transformer import Transformer as BitNetTransformerBlock, RMSNorm as BitNetRMSNorm

	# override because bitnet's BitNetTransformer includes an embedding input / classifier output layers inside of it, which isn't favorable
	class BitNetTransformer(nn.Module):
		def __init__(
			self,
			dim: int,
			depth: int,
			num_tokens: int,
			heads=8,
			ff_mult=4,
		):
			super().__init__()

			self.transformer = BitNetTransformerBlock( dim=dim, depth=depth, heads=heads, ff_mult=ff_mult )
			self.norm = BitNetRMSNorm(dim)

		def forward(self, x):
			x = self.transformer(x)
			return self.norm( x )

	"""
	from bitnet import BitNetTransformer
	def NoEmbedding_BitNetTransformer_Forward(self, x):
		x = self.transformer(x)
		return self.to_logits[0](x)

	BitNetTransformer.forward = NoEmbedding_BitNetTransformer_Forward 
	"""

except Exception as e:
	print("Error importing `bitnet` arch:", e)
	pass

try:
	from transformers import MixtralModel, MixtralConfig
	from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func, MixtralSparseMoeBlock

	# This is required because batch sizes > 1 throws errors
	def Fixed_MixtralSparseMoeBlock_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		""" """
		batch_size, sequence_length, hidden_dim = hidden_states.shape
		hidden_states = hidden_states.reshape(-1, hidden_dim) # was view()
		# router_logits: (batch * sequence_length, n_experts)
		router_logits = self.gate(hidden_states)

		routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
		routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
		routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
		# we cast back to the input dtype
		routing_weights = routing_weights.to(hidden_states.dtype)

		final_hidden_states = torch.zeros(
			(batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
		)

		expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

		for expert_idx in range(self.num_experts):
			expert_layer = self.experts[expert_idx]
			idx, top_x = torch.where(expert_mask[expert_idx])

			if top_x.shape[0] == 0:
				continue
			top_x_list = top_x.tolist()
			idx_list = idx.tolist()

			current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
			current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

			final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
		final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
		return final_hidden_states, router_logits

	Original_MixtralSparseMoeBlock_forward = MixtralSparseMoeBlock.forward
	MixtralSparseMoeBlock.forward = Fixed_MixtralSparseMoeBlock_forward

except Exception as e:
	print("Error importing `mixtral` arch:", e)


AVAILABLE_ATTENTIONS = ["mem_efficient", "math"]

try:
	from xformers.ops import LowerTriangularMask
	from xformers.ops.fmha import memory_efficient_attention

	AVAILABLE_ATTENTIONS.append("xformers")
except Exception as e:
	print("Error while importing `xformers`", e)

try:
	from transformers.utils import is_flash_attn_2_available

	if is_flash_attn_2_available():
		AVAILABLE_ATTENTIONS.append("flash")
except Exception as e:
	raise e

try:
	from transformers.cache_utils import Cache
	from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb


	class Llama_Attention(LlamaAttention):
		def __init__(self, *args, **kwargs):
			if 'mode' in kwargs:
				self.mode = kwargs['mode']
				kwargs.pop("mode")
			else:
				self.mode = "math"

			super().__init__(*args, **kwargs)

		def forward(
			self,
			hidden_states: torch.Tensor,
			attention_mask: Optional[torch.Tensor] = None,
			position_ids: Optional[torch.LongTensor] = None,
			past_key_value: Optional[Cache] = None,
			output_attentions: bool = False,
			use_cache: bool = False,
			cache_position: Optional[torch.LongTensor] = None,
			**kwargs,
		) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
			bsz, q_len, _ = hidden_states.size()

			query_states = self.q_proj(hidden_states)
			key_states = self.k_proj(hidden_states)
			value_states = self.v_proj(hidden_states)

			query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
			key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
			value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

			cos, sin = self.rotary_emb(value_states, position_ids)
			query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

			past_key_value = getattr(self, "past_key_value", past_key_value)

			if past_key_value is not None:
				# sin and cos are specific to RoPE models; cache_position needed for the static cache
				cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
				key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

			query_states = query_states.transpose(1, 2)
			key_states = key_states.transpose(1, 2)
			value_states = value_states.transpose(1, 2)

			dropout_rate = self.attention_dropout if self.training else 0.0

			if self.mode == "xformers":
				if attention_mask is None or attention_mask[0, 0, 0, 1] == 0:
					attn_output = memory_efficient_attention(query_states, key_states, value_states, attn_bias=None)
				else:
					attn_output = memory_efficient_attention(query_states, key_states, value_states, attn_bias=LowerTriangularMask())
			else:
				with torch.backends.cuda.sdp_kernel(enable_flash=self.mode == "flash", enable_math=self.mode == "math", enable_mem_efficient=self.mode == "mem_efficient"):
					attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask)

			attn_weights = None

			attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
			attn_output = self.o_proj(attn_output)

			return attn_output, attn_weights, past_key_value
except Exception as e:
	print("Error creating modified `LLamaAttention`:", e)

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
class AudioEmbedding(nn.Module):
	def __init__(
		self,
		l_tokens: int, # list of number of tokens (needed because AR resps includes stop token)
		token_dim: int, # dimensionality of the embedding
		levels: int | None = None, # number of RVQ-bins (I don't remember the specifics)
		sums: bool = True # whether to sum all previous layers of embeddings to factor in other RVQ bin levels (I do not know which way is better)
	):
		super().__init__()
		# array of embeddings
		#   proms are [0, prom_levels]
		#   resp are split to where [0] is for the AR, and [1:] are reserved for NAR
		self.embeddings = nn.ModuleList([nn.Embedding(n_tokens, token_dim) for n_tokens in l_tokens])
		# weight influencer for the influence for each level (desu this should be really useless because the weights in the embedding themselves should factor this)
		self.weight = nn.ParameterList([nn.Parameter( torch.Tensor([1]) ) for i in range(levels)]) if levels is not None else None
		# 
		self.sums = sums
	
	def forward(self, xi: Tensor, quant_levels: Tensor | None = None ) -> Tensor:
		# prom
		if quant_levels is None and xi.shape[-1] > 1:
			if self.sums:
				x = sum( [ self.embeddings[k]( xi[:, k] ) * (self.weight[k] if self.weight is not None else 1) for k in range(xi.shape[-1]) ] )
			else:
				k = 0 # only use the most significant RVQ bin level for the input prom
				x = self.embeddings[k]( xi[:, k] ) * (self.weight[k] if self.weight is not None else 1)
		# AR resp
		elif quant_levels is None or quant_levels == 0:
			x = self.embeddings[0]( xi[:, 0] )
		# NAR resp
		else:
			if self.sums:
				x = sum( [ self.embeddings[k+1]( xi[:, k] ) * (self.weight[k+1] if self.weight is not None else 1) for k in range(xi.shape[-1]) ] )
			else:
				k = xi.shape[-1] - 1 # only use the previous RVQ bin level for the current resp embedding
				x = self.embeddings[k+1]( xi[:, k] ) * (self.weight[k+1] if self.weight is not None else 1)
		
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
	def recurrent_chunk_size(self) -> int:
		raise NotImplementedError

	@property
	def rotary_embedding_base(self) -> float:
		return 10000
	
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
		return self.n_tokens

	@property
	def ignore_index(self):
		return -100

	def __init__(
		self,
		n_tokens: int = 1024,
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
		self.activation_checkpointing = self.config.activation_checkpointing if self.config is not None else True

		self.n_tokens = n_tokens
		self.d_model = d_model
		self.n_heads = n_heads
		self.n_layers = n_layers
		self.n_experts = n_experts
		
		self.l_padding = l_padding

		# +1 to include the stop token
		# to-do: undo this dogshit mistake; tasks tokens should be delegated to its own embedding
		n_prom_tokens = n_tokens
		n_resp_tokens = n_tokens + (1 if self.causal else 0) # AR requires a stop token to... know when to stop

		self.text_emb = Embedding(n_tokens, d_model)
		self.langs_emb = None
		self.tones_emb = None
		self.tasks_emb = None

		if self.version == 1: # legacy
			n_prom_tokens += (self.n_tasks - 1) # old models have the task tokens in the prom
			self.proms_emb = MultiEmbedding(self.n_prom_levels, n_prom_tokens, d_model)
			self.resps_emb = MultiEmbedding(self.n_resp_levels, n_resp_tokens, d_model, monolithic=self.monolithic)
		else:
			# [1024] * 8
			self.proms_emb = AudioEmbedding(
				[n_prom_tokens] * self.n_prom_levels, d_model,
				levels=self.n_prom_levels if self.version > 3 else None,
				sums=self.config.audio_embedding_sums if self.config is not None else True,
			)
			# [1025] + [1024] * 8
			self.resps_emb = AudioEmbedding(
				[n_resp_tokens] + [n_resp_tokens - 1] * (self.n_resp_levels - 1), d_model,
				levels=self.n_resp_levels if self.version > 3 else None,
				sums=self.config.audio_embedding_sums if self.config is not None else True
			)

		
		if self.version >= 3:
			self.langs_emb = Embedding(self.n_langs, d_model) if self.n_langs > 0 else None
			self.tasks_emb = Embedding(self.n_tasks, d_model) if self.n_tasks > 0 else None
		
		if self.version >= 4:
			self.tones_emb = Embedding(self.n_tones, d_model) if self.n_tones > 0 else None

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
					num_key_value_heads=n_heads,
					hidden_act="gelu",
					is_encoder_decoder=False,
					is_decoder=True,
					attn_implementation=hf_attention,
					#gradient_checkpointing=self.activation_checkpointing,
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
					#gradient_checkpointing=self.activation_checkpointing,
				))

			if self.activation_checkpointing and not self.model.gradient_checkpointing:
				self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(
					use_reentrant=False
				))

			if training:
				self.model.training = True
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
					#gradient_checkpointing=self.activation_checkpointing,
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
					#gradient_checkpointing=self.activation_checkpointing,
				))

			if self.activation_checkpointing and not self.model.gradient_checkpointing:
				self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(
					use_reentrant=False
				))

			if training:
				self.model.training = True
		elif self.arch_type == "retnet":
			kwargs = dict(
				vocab_size=n_resp_tokens,
				decoder_embed_dim=d_model,
				decoder_value_embed_dim =d_model * 2,
				decoder_retention_heads=n_heads,
				decoder_ffn_embed_dim=d_model * 4,
				decoder_layers=n_layers,
				dropout=p_dropout if training else 0.0,
				checkpoint_activations=self.activation_checkpointing,
				activation_fn="gelu",
				use_layernorm=self.version < 3,
				use_biases=self.version < 3,
				use_glu=self.version >= 3,

				chunkwise_recurrent=self.causal and self.recurrent_chunk_size > 0,
				recurrent_chunkwise_size=self.recurrent_chunk_size if self.causal else 0,
				no_output_layer=True,
				decoder_normalize_before=True,

				rotary_embedding_base=self.rotary_embedding_base, # 10000
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
				checkpoint_activations=self.activation_checkpointing,
				activation_fn="gelu",
				use_glu=False, # self.version >= 3,

				recurrent_chunk_size=self.recurrent_chunk_size if self.causal else 0,
				decoder_normalize_before=True,

				deepnorm=False,
				subln=True,
			)

			self.model = RetNetDecoder_HF(RetNetConfig_HF(**kwargs))
		elif self.arch_type == "bitnet":
			self.model = BitNetTransformer(
				num_tokens=n_resp_tokens,
				dim=d_model,
				depth=n_layers,
				heads=n_heads,
				ff_mult=4,
			)
		else:
			raise RuntimeError(f'Unknown arch specified: {self.arch_type}')

		if self.config.attention in ["xformers", "auto", "mem_efficient", "math", "flash"]:
			self.model = ml.replace_attention( self.model, klass=Llama_Attention, target=LlamaAttention, mode=self.config.attention )

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

		"""
		# Broken
		if state is not None and (self.arch_type == "retnet" or self.arch_type == "retnet-hf"):
			# prefill
			if len(state) == 0:
				prefill_size = x.shape[1]
				# run the initial prompt to fill the KV cache
				if self.arch_type == "retnet":
					for n in range(prefill_size):
						xi = x[:, n, :].unsqueeze(1)
						self.model(xi, incremental_state=state, token_embeddings=xi, features_only=True)
				elif self.arch_type == "retnet-hf":
					state = None
					for n in range(prefill_size):
						xi = x[:, n, :].unsqueeze(1)

						kwargs = dict(
							attention_mask=m,
							inputs_embeds=xi,
							past_key_values=state,
							use_cache=True,
							forward_impl='recurrent',
						#	return_dict=True,
						)

						out = self.model(**kwargs)
						state = out.past_key_values

			# grab last token(s)
			x = x[:, -1, :].unsqueeze(1)
		"""
		
		# HF transformer derived model
		if self.arch_type in ["llama", "mistral", "mixtral"]:
			kwargs = dict(
				attention_mask=m,
				inputs_embeds=x,
				past_key_values=state,
				use_cache=True,
			#	return_dict=True,
			)
			if self.n_experts > 1 and targ_list is not None:
				kwargs["output_router_logits"] = True

			t = self.model(**kwargs)

			x = t[0]
			
			if state is not None:
				state = t[1]
			
			if self.n_experts > 1 and targ_list is not None:
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
		targ_list: list[Tensor] | None = None,

		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
	):
		device = text_list[0].device
		batch_size = len(text_list)

		inputs = [ [] for _ in range(batch_size) ]
		for i in range(batch_size):
			if text_list is not None:
				inputs[i].append( ( "text", text_list[i] ) )
			if proms_list is not None:
				inputs[i].append( ( "prom", proms_list[i] ) )
			if resps_list is not None:
				inputs[i].append( ( "resp", resps_list[i] ) )
			if targ_list is not None:
				inputs[i].append( ( "targ", targ_list[i] ) )

		return inputs

	def inputs_to_embeddings(
		self,
		inputs: list,
		quant_levels: Tensor | None = None
	):
		x_list = []
		for b_i in range(len(inputs)):
			batch = []
			for i in range(len(inputs[b_i])):
				name, input = inputs[b_i][i]
				embedding = None
				if name == "text":
					embedding = self.text_emb( input )
				elif name == "lang":
					embedding = self.langs_emb( input )
				elif name == "prom":
					embedding = self.proms_emb( input )
				elif name == "tone":
					embedding = self.tones_emb( input )
				elif name == "resp":
					embedding = self.resps_emb( input, quant_levels[b_i] if quant_levels is not None else None )
				else:
					continue

				batch.append(embedding)
	
			x_list.append( _join( batch, self.sep ) )

		return x_list

	def training_targets(
		self,
		inputs: list,
	):
		x_list = []
		for bi in range(len(inputs)):
			batch = []
			for i in range(len(inputs[bi])):
				name, input = inputs[bi][i]
				device = input.device

				if name == "prom":
					batch.append( torch.full_like(input[..., 0], self.ignore_index) )
				elif name in ["text", "lang", "tone", "targ"]:
					batch.append( input )

			x_list.append( _join( batch, torch.tensor(self.ignore_index, device=device) ) )

		return x_list

	def forward(
		self,
		inputs: list,

		quant_levels: Tensor | None = None,
		state: dict | list | None = None,
	):

		x_list = self.inputs_to_embeddings( inputs, quant_levels )
		x, m = list_to_tensor(x_list)

		# yes, there's a better way.
		training = False
		for b_i in range(len(inputs)):
			for i in range(len(inputs[b_i])):
				name, input = inputs[b_i][i]
				if name == "targ":
					training = True


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
			target_list = self.training_targets( inputs )
			
			# modify only for the AR so it can properly behave like a transformer
			for i in range(len(target_list)):
				if quant_levels is not None and quant_levels[i] > 0:
					continue

				logits[i] = logits[i][..., :-1, :] # shift the target so that token n...
				target_list[i] = target_list[i][..., 1:] # predicts token n + 1

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

			if aux_loss is not None:
				self.loss["nll"] += aux_loss
			
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
		if quant_levels is not None:
			logits = [ logit[-l:] for logit, l in zip(logits, map(len, resps_list)) ]
		# (AR chunkwise) return the last chunkwise piece
		elif self.causal and self.recurrent_chunk_size > 0:
			logits = [ logit[-l:] for logit, l in zip(logits, self.recurrent_chunk_size) ]
		# (AR) return just the last code
		else:
			logits = [ logit[-1:] for logit in logits ]

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