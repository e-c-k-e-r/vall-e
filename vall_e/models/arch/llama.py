import math
import torch
import logging
import random
from typing import Literal, overload, Optional, Tuple, Union, List

from torch import Tensor, nn

# lazy
from transformers.models.llama.configuration_llama import LlamaConfig as BaseConfig
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.activations import ACT2FN

from .attention import *

class Config(BaseConfig):
	def __init__(
		self,
		attn_mode = "sdpa",
		output_norm = True,
		*args, **kwargs
	):
		super().__init__(*args, **kwargs)

		self.attn_mode = attn_mode
		self.output_norm = output_norm

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
	batch, num_key_value_heads, slen, head_dim = hidden_states.shape

	if n_rep == 1:
		return hidden_states

	hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)

	return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
	x1 = x[..., : x.shape[-1] // 2]
	x2 = x[..., x.shape[-1] // 2 :]

	return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
	cos = cos.unsqueeze(unsqueeze_dim)
	sin = sin.unsqueeze(unsqueeze_dim)
	q_embed = (q * cos) + (rotate_half(q) * sin)
	k_embed = (k * cos) + (rotate_half(k) * sin)

	return q_embed, k_embed


class RMSNorm(nn.Module):
	def __init__(self, hidden_size, eps=1e-6):
		super().__init__()

		self.weight = nn.Parameter(torch.ones(hidden_size))
		self.variance_epsilon = eps

	def forward(self, hidden_states):
		input_dtype = hidden_states.dtype
		hidden_states = hidden_states.to(torch.float32)
		variance = hidden_states.pow(2).mean(-1, keepdim=True)
		hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
		
		return self.weight * hidden_states.to(input_dtype)

	def extra_repr(self):
		return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class RotaryEmbedding(nn.Module):
	def __init__(self, config, device=None):
		super().__init__()

		if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
			self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
		else:
			self.rope_type = "default"

		self.max_seq_len_cached = config.max_position_embeddings
		self.original_max_seq_len = config.max_position_embeddings

		self.config = config
		self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

		inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
		self.register_buffer("inv_freq", inv_freq, persistent=False)
		self.original_inv_freq = self.inv_freq

	def _dynamic_frequency_update(self, position_ids, device):
		seq_len = torch.max(position_ids) + 1
		if seq_len > self.max_seq_len_cached:  # growth
			inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
			self.register_buffer("inv_freq", inv_freq, persistent=False) 
			self.max_seq_len_cached = seq_len

		if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:
			self.original_inv_freq = self.original_inv_freq.to(device)
			self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
			self.max_seq_len_cached = self.original_max_seq_len

	@torch.no_grad()
	def forward(self, x, position_ids):
		if "dynamic" in self.rope_type:
			self._dynamic_frequency_update(position_ids, device=x.device)

		inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
		position_ids_expanded = position_ids[:, None, :].float()

		device_type = x.device.type
		device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
		with torch.autocast(device_type=device_type, enabled=False):
			freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(1, 2)
			emb = torch.cat((freqs, freqs), dim=-1)
			cos = emb.cos()
			sin = emb.sin()

		cos = cos * self.attention_scaling
		sin = sin * self.attention_scaling

		return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class Attention(nn.Module):
	def __init__(self, config, layer_idx):
		super().__init__()

		self.config = config
		self.attn_mode = config.attn_mode
		self.layer_idx = layer_idx
		self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
		self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
		self.scaling = self.head_dim**-0.5
		self.attention_dropout = config.attention_dropout
		self.num_heads = config.num_attention_heads
		self.num_key_value_heads = config.num_key_value_heads

		if self.attn_mode == "math":
			self.attn_mode = torch.nn.attention.SDPBackend.MATH
		elif self.attn_mode == "mem_efficient":
			self.attn_mode = torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
		elif self.attn_mode == "flash_(sdpa)":
			self.attn_mode = torch.nn.attention.SDPBackend.FLASH_ATTENTION
		elif self.attn_mode == "cudnn":
			self.attn_mode = torch.nn.attention.SDPBackend.CUDNN_ATTENTION
		elif self.attn_mode == "sdpa":
			self.attn_mode = torch.nn.attention.SDPBackend.MATH

		self.q_proj = nn.Linear(
			config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
		)
		self.k_proj = nn.Linear(
			config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
		)
		self.v_proj = nn.Linear(
			config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
		)
		self.o_proj = nn.Linear(
			config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
		)

	# extracts inputs from a batch based on requested causality
	def split_forward(
		self,
		hidden_states: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
		is_causal: Optional[list] = None,
		target_causal_state: Optional[bool] = True,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_value: Optional[Cache] = None,
		output_attentions: bool = False,
		use_cache: bool = False,
		cache_position: Optional[torch.LongTensor] = None,
		position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
		**kwargs,
	):
		indices = [ i for i, state in enumerate( is_causal ) if state == target_causal_state ]

		# no matching inputs in batch
		if not indices:
			return indices, None, None, None

		# entire batch is homogenous
		if len( indices ) == hidden_states.shape[0]:
			output_hidden_states, output_self_attn_weights, output_present_key_values = self.forward(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				is_causal=target_causal_state,
				position_ids=position_ids,
				past_key_value=past_key_value,
				output_attentions=output_attentions,
				use_cache=False,
				cache_position=cache_position,
				position_embeddings=position_embeddings,
				**kwargs,
			)
			return indices, output_hidden_states, output_self_attn_weights, output_present_key_values

		input_hidden_states = torch.stack( [ hidden_states[i] for i in indices ] )
		input_attention_mask = torch.stack( [ attention_mask[i] for i in indices ] ) if attention_mask is not None else None
		input_position_ids = torch.stack( [ position_ids[i] for i in indices ] ) if position_ids is not None else None
		input_position_embeddings = (
			torch.stack( [ position_embeddings[0][i] for i in indices ] ),
			torch.stack( [ position_embeddings[1][i] for i in indices ] ),
		) if position_embeddings is not None else None

		output_hidden_states, output_self_attn_weights, output_present_key_values = self.forward(
			hidden_states=input_hidden_states,
			attention_mask=input_attention_mask,
			is_causal=target_causal_state,
			position_ids=input_position_ids,
			past_key_value=past_key_value,
			output_attentions=output_attentions,
			use_cache=False,
			cache_position=cache_position,
			position_embeddings=input_position_embeddings,
			**kwargs,
		)
		return indices, output_hidden_states, output_self_attn_weights, output_present_key_values

	# Adapted from LlamaAttention.forward
	def forward(
		self,
		hidden_states: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
		is_causal: bool = True,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_value: Optional[Cache] = None,
		output_attentions: bool = False,
		use_cache: bool = False,
		cache_position: Optional[torch.LongTensor] = None,
		position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
		**kwargs,
	) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
		mode = "default" if output_attentions else self.attn_mode
		non_split_attention = [
			"default",
			torch.nn.attention.SDPBackend.MATH,
			torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
			torch.nn.attention.SDPBackend.FLASH_ATTENTION,
			torch.nn.attention.SDPBackend.CUDNN_ATTENTION
		]

		# split per batch because other attention mechanisms do not have a conditional is_causal per-batch, only for the entire input
		if isinstance( is_causal, list ) and mode not in non_split_attention:
			# initialize lists
			attn_hidden_states = [ None for _ in is_causal ]
			self_attn_weights = [ None for _ in is_causal ]
			present_key_values = [ None for _ in is_causal ]

			# process causal inputs in a batch
			causal_indices, causal_hidden_states, causal_self_attn_weights, causal_present_key_values = self.split_forward(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				is_causal=is_causal,
				target_causal_state=True,
				position_ids=position_ids,
				past_key_value=past_key_value,
				output_attentions=output_attentions,
				use_cache=False,
				cache_position=cache_position,
				position_embeddings=position_embeddings,
				**kwargs,
			)

			# process non-causal inputs in a batch
			non_causal_indices, non_causal_hidden_states, non_causal_self_attn_weights, non_causal_present_key_values = self.split_forward(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				is_causal=is_causal,
				target_causal_state=False,
				position_ids=position_ids,
				past_key_value=past_key_value,
				output_attentions=output_attentions,
				use_cache=False,
				cache_position=cache_position,
				position_embeddings=position_embeddings,
				**kwargs,
			)

			# insert causal outputs to batch
			for i, idx in enumerate( causal_indices ):
				attn_hidden_states[idx] = causal_hidden_states[i]

				if output_attentions:
					self_attn_weights[idx] = causal_self_attn_weights[i]

			# insert non-causal outputs to batch
			for i, idx in enumerate( non_causal_indices ):
				attn_hidden_states[idx] = non_causal_hidden_states[i]

				if output_attentions:
					self_attn_weights[idx] = non_causal_self_attn_weights[i]

			# combine list
			attn_hidden_states = torch.stack( attn_hidden_states, dim=0 )
			if output_attentions:
				self_attn_weights = torch.stack( self_attn_weights, dim=0 )

			return attn_hidden_states, output_attentions, []

		dropout_rate = self.attention_dropout if self.training else 0.0
		bsz, q_len, _ = hidden_states.size()

		query_states = self.q_proj(hidden_states)
		key_states = self.k_proj(hidden_states)
		value_states = self.v_proj(hidden_states)

		query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
		key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
		value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

		if position_embeddings is None:
			cos, sin = self.rotary_emb(value_states, position_ids)
		else:
			cos, sin = position_embeddings
		query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

		if past_key_value is not None:
			# sin and cos are specific to RoPE models; cache_position needed for the static cache
			cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
			key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

		attn_scores = None

		if mode in ["xformers", "flash_attn"]:
			query_states = query_states.transpose(1, 2)
			key_states = key_states.transpose(1, 2)
			value_states = value_states.transpose(1, 2)

			if mode == "flash_attn":
				attn_output = flash_attn_func(
					query_states,
					key_states,
					value_states,
					causal=is_causal,
					softmax_scale=1.0 / math.sqrt(self.head_dim),
					dropout_p=dropout_rate,
				)
				
				attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
			elif mode == "xformers":
				attn_output = memory_efficient_attention(
					query_states,
					key_states,
					value_states,
					attn_bias = LowerTriangularMask(),
					scale = 1.0 / math.sqrt(self.head_dim),
					p=dropout_rate
				)
				attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

			attn_output = self.o_proj(attn_output)
			return attn_output, attn_scores, past_key_value

		key_states = repeat_kv(key_states, self.num_key_value_groups)
		value_states = repeat_kv(value_states, self.num_key_value_groups)

		x_mask = attention_mask
		
		if attention_mask is not None:
			x_mask = x_mask[:, :, :, : key_states.shape[-2]]

		# SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
		# Reference: https://github.com/pytorch/pytorch/issues/112577.
		if query_states.device.type == "cuda" and x_mask is not None:
			query_states = query_states.contiguous()
			key_states = key_states.contiguous()
			value_states = value_states.contiguous()

		if mode in ["sageattn"]:
			attn_output = sageattn(
				query_states,
				key_states,
				value_states,
				tensor_layout="HND",
				is_causal=is_causal
			)
		elif mode in ["fused_attn"]:
			attn_output = fused_attn_func(
				query_states,
				key_states,
				value_states,
				causal=is_causal,
				softmax_scale=1.0 / math.sqrt(self.head_dim),
				dropout_p=dropout_rate,
			)
		elif mode in ["default"]:
			attn_scores = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
			# cringe logic
			attn_weights = (attn_scores + x_mask) if attention_mask is not None else (attn_scores)
			# upcast attention to fp32
			attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
			attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
			attn_output = torch.matmul(attn_weights, value_states)

			if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
				raise ValueError(
					f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
					f" {attn_output.size()}"
				)
		else:
			# We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
			# in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
			# is_causal = True if x_mask is None and q_len > 1 else False
			is_causal = True if x_mask is None and q_len > 1 else False
			with torch.nn.attention.sdpa_kernel(self.attn_mode):
				attn_output = torch.nn.functional.scaled_dot_product_attention(
					query_states,
					key_states,
					value_states,
					attn_mask=x_mask,
					dropout_p=dropout_rate,
					is_causal=is_causal,
				)

		# cringe
		if attn_scores is None and output_attentions:
			attn_scores = attn_output

		attn_output = attn_output.transpose(1, 2).contiguous()
		attn_output = attn_output.view(bsz, q_len, -1)

		attn_output = self.o_proj(attn_output)

		return attn_output, attn_scores, past_key_value

class MLP(nn.Module):
	def __init__(self, config):
		super().__init__()

		self.config = config
		self.hidden_size = config.hidden_size
		self.intermediate_size = config.intermediate_size
		self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
		self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
		self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
		self.act_fn = ACT2FN[config.hidden_act]

	def forward(self, x):
		down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
		return down_proj

class DecoderLayer(nn.Module):
	def __init__(self, config, layer_idx):
		super().__init__()
	
		self.hidden_size = config.hidden_size

		self.self_attn = Attention(config=config, layer_idx=layer_idx)

		self.mlp = MLP(config)
		self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
		self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

	def forward(
		self,
		hidden_states: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
		is_causal: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_value: Optional[Cache] = None,
		output_attentions: Optional[bool] = False,
		use_cache: Optional[bool] = False,
		cache_position: Optional[torch.LongTensor] = None,
		position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
		**kwargs,
	) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
		residual = hidden_states

		hidden_states = self.input_layernorm(hidden_states)
		
		# ugh
		if isinstance( is_causal, list ) and len(is_causal) == 1:
			is_causal = is_causal[0]

		# Self Attention
		hidden_states, self_attn_weights, present_key_value = self.self_attn(
			hidden_states=hidden_states,
			attention_mask=attention_mask,
			is_causal=is_causal,
			position_ids=position_ids,
			past_key_value=past_key_value,
			output_attentions=output_attentions,
			use_cache=use_cache,
			cache_position=cache_position,
			position_embeddings=position_embeddings,
			**kwargs,
		)
		hidden_states = residual + hidden_states

		# Fully Connected
		residual = hidden_states
		hidden_states = self.post_attention_layernorm(hidden_states)
		hidden_states = self.mlp(hidden_states)
		hidden_states = residual + hidden_states

		outputs = (hidden_states,)

		if output_attentions:
			outputs += (self_attn_weights,)

		if use_cache:
			outputs += (present_key_value,)

		return outputs

class Model(LlamaPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)

		self.padding_idx = config.pad_token_id
		self.vocab_size = config.vocab_size
		self.layers_n = config.num_hidden_layers

		self.layers = nn.ModuleList(
			[DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
		)
		self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.output_norm else nn.Identity()
		self.rotary_emb = RotaryEmbedding(config=config)
		self.gradient_checkpointing = False

		# Initialize weights and apply final processing
		self.post_init()

	# shamelessly borrowed from https://github.com/open-mmlab/Amphion/blob/main/models/tts/maskgct/llama_nar.py#L256 until I replace it with my own noncausal-mask maker
	def _update_noncausal_mask(
		self,
		attention_mask,
		inputs_embeds,
		past_key_values_length,
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

	@staticmethod
	def _prepare_4d_causal_attention_mask_with_cache_position(
		attention_mask: torch.Tensor,
		sequence_length: int,
		target_length: int,
		dtype: torch.dtype,
		device: torch.device,
		cache_position: torch.Tensor,
		batch_size: int,
		**kwargs,
	):
		if attention_mask is not None and attention_mask.dim() == 4:
			causal_mask = attention_mask
		else:
			min_dtype = torch.finfo(dtype).min
			causal_mask = torch.full(
				(sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
			)
			if sequence_length != 1:
				causal_mask = torch.triu(causal_mask, diagonal=1)
			causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
			causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
			if attention_mask is not None:
				causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
				mask_length = attention_mask.shape[-1]
				padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
					causal_mask.device
				)
				padding_mask = padding_mask == 0
				causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
					padding_mask, min_dtype
				)

		return causal_mask

	# gut out the things that just shoves responsibility on SDPA's is_causal generating a mask because this causes problems
	def _update_causal_mask(
		self,
		attention_mask: torch.Tensor,
		input_tensor: torch.Tensor,
		cache_position: torch.Tensor,
		past_key_values: Cache,
		output_attentions: bool,
	):
		past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
		using_static_cache = isinstance(past_key_values, StaticCache)

		dtype, device = input_tensor.dtype, input_tensor.device
		sequence_length = input_tensor.shape[1]
		if using_static_cache:
			target_length = past_key_values.get_max_cache_shape()
		else:
			target_length = (
				attention_mask.shape[-1]
				if isinstance(attention_mask, torch.Tensor)
				else past_seen_tokens + sequence_length + 1
			)

		# In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
		causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
			attention_mask,
			sequence_length=sequence_length,
			target_length=target_length,
			dtype=dtype,
			device=device,
			cache_position=cache_position,
			batch_size=input_tensor.shape[0],
		)

		if (
			self.config._attn_implementation == "sdpa"
			and attention_mask is not None
			and attention_mask.device.type == "cuda"
			and not output_attentions
		):
			min_dtype = torch.finfo(dtype).min
			causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

		return causal_mask

	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		is_causal: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		cache_position: Optional[torch.LongTensor] = None,		
	) -> Union[Tuple, BaseModelOutputWithPast]:
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		use_cache = use_cache if use_cache is not None else self.config.use_cache
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

		if self.gradient_checkpointing and self.training and use_cache:
			_logger.warning_once(
				"`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
			)
			use_cache = False

		# kept for BC (non `Cache` `past_key_values` inputs)
		return_legacy_cache = False
		if use_cache and not isinstance(past_key_values, Cache):
			return_legacy_cache = True
			if past_key_values is None:
				past_key_values = DynamicCache()
			else:
				past_key_values = DynamicCache.from_legacy_cache(past_key_values)
				_logger.warning_once(
					"We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
					"will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
					"(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
				)

		if cache_position is None:
			past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
			cache_position = torch.arange(
				past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
			)
		if position_ids is None:
			position_ids = cache_position.unsqueeze(0)

		# because we can attend to both a causal and a non-causal sequence, generate both masks then pick among which to use per batch
		if is_causal is not None:
			causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)
			noncausal_mask = self._update_noncausal_mask(attention_mask, inputs_embeds, past_key_values)

			x_mask = torch.stack( [ causal_mask[i, :, :, :] if state else noncausal_mask[i, :, :, :] for i, state in enumerate( is_causal ) ], dim=0 )
		else:
			x_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)

		hidden_states = inputs_embeds

		# create position embeddings to be shared across the decoder layers
		position_embeddings = self.rotary_emb(hidden_states, position_ids)

		# decoder layers
		all_hidden_states = () if output_hidden_states else None
		all_self_attns = () if output_attentions else None
		next_decoder_cache = None

		for l, decoder_layer in enumerate(self.layers):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			if self.gradient_checkpointing and self.training:
				layer_outputs = self._gradient_checkpointing_func(
					decoder_layer.__call__,
					hidden_states,
					x_mask,
					is_causal,
					position_ids,
					past_key_values,
					output_attentions,
					use_cache,
					cache_position,
					position_embeddings,
				)
			else:
				layer_outputs = decoder_layer(
					hidden_states,
					attention_mask=x_mask,
					is_causal=is_causal,
					position_ids=position_ids,
					past_key_value=past_key_values,
					output_attentions=output_attentions,
					use_cache=use_cache,
					cache_position=cache_position,
					position_embeddings=position_embeddings,
				)

			hidden_states = layer_outputs[0]

			if use_cache:
				next_decoder_cache = layer_outputs[2 if output_attentions else 1]

			if output_attentions:
				all_self_attns += (layer_outputs[1],)

		hidden_states = self.norm(hidden_states)

		# add hidden states from the last decoder layer
		if output_hidden_states:
			all_hidden_states += (hidden_states,)

		next_cache = next_decoder_cache if use_cache else None
		if return_legacy_cache:
			next_cache = next_cache.to_legacy_cache()

		if not return_dict:
			return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

		return BaseModelOutputWithPast(
			last_hidden_state=hidden_states,
			past_key_values=next_cache,
			hidden_states=all_hidden_states,
			attentions=all_self_attns,
		)