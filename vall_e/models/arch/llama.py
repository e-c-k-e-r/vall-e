# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

import torch
from typing import Literal, overload, Optional, Tuple

from torch import Tensor, nn
from transformers.cache_utils import Cache

from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv

AVAILABLE_ATTENTIONS = ["sdpa"]

if torch.backends.cuda.flash_sdp_enabled():
	AVAILABLE_ATTENTIONS.append("flash")	

if torch.backends.cuda.mem_efficient_sdp_enabled():
	AVAILABLE_ATTENTIONS.append("mem_efficient")	

if torch.backends.cuda.math_sdp_enabled():
	AVAILABLE_ATTENTIONS.append("math")	

try:
	from xformers.ops import LowerTriangularMask
	from xformers.ops.fmha import memory_efficient_attention

	AVAILABLE_ATTENTIONS.append("xformers")
except Exception as e:
	print("Error while importing `xformers`", e)

try:
	from transformers.utils import is_flash_attn_2_available

	if is_flash_attn_2_available():
		AVAILABLE_ATTENTIONS.append("flash_attention_2")
except Exception as e:
	print("Error while querying for `flash_attn_2` support", e)

class LlamaAttention_Adapted(LlamaAttention):
	def __init__(self, *args, **kwargs):
		if 'mode' in kwargs:
			self.mode = kwargs['mode']
			kwargs.pop("mode")
		else:
			self.mode = "math"

		if self.mode == "math":
			self.mode = torch.nn.attention.SDPBackend.MATH
		elif self.mode == "mem_efficient":
			self.mode = torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
		elif self.mode == "flash":
			self.mode = torch.nn.attention.SDPBackend.FLASH_ATTENTION
		elif self.mode == "cudnn":
			self.mode = torch.nn.attention.SDPBackend.CUDNN_ATTENTION

		super().__init__(*args, **kwargs)

	 # Adapted from LlamaAttention.forward
	def forward(
		self,
		hidden_states: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_value: Optional[Cache] = None,
		output_attentions: bool = False,
		use_cache: bool = False,
		cache_position: Optional[torch.LongTensor] = None,
		position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
		**kwargs,
	) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
		if output_attentions:
			# TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
			return super().forward(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				past_key_value=past_key_value,
				output_attentions=output_attentions,
				use_cache=use_cache,
				cache_position=cache_position,
				position_embeddings=position_embeddings,
			)

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

		key_states = repeat_kv(key_states, self.num_key_value_groups)
		value_states = repeat_kv(value_states, self.num_key_value_groups)

		causal_mask = attention_mask
		if attention_mask is not None:
			causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

		# SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
		# Reference: https://github.com/pytorch/pytorch/issues/112577.
		if query_states.device.type == "cuda" and causal_mask is not None:
			query_states = query_states.contiguous()
			key_states = key_states.contiguous()
			value_states = value_states.contiguous()

		# We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
		# in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
		is_causal = True if causal_mask is None and q_len > 1 else False

		#with torch.backends.cuda.sdp_kernel(enable_flash=self.mode == "flash", enable_math=self.mode == "math", enable_mem_efficient=self.mode == "mem_efficient"):
		with torch.nn.attention.sdpa_kernel(self.mode):
			attn_output = torch.nn.functional.scaled_dot_product_attention(
				query_states,
				key_states,
				value_states,
				attn_mask=causal_mask,
				dropout_p=self.attention_dropout if self.training else 0.0,
				is_causal=is_causal,
			)

		attn_output = attn_output.transpose(1, 2).contiguous()
		attn_output = attn_output.view(bsz, q_len, -1)

		attn_output = self.o_proj(attn_output)

		return attn_output, None, past_key_value

	"""
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
				attn_output = memory_efficient_attention(query_states, key_states, value_states, attn_bias=None, p=dropout_rate)
			else:
				attn_output = memory_efficient_attention(query_states, key_states, value_states, attn_bias=LowerTriangularMask(), p=dropout_rate)
		else:
			with torch.backends.cuda.sdp_kernel(enable_flash=self.mode == "flash", enable_math=self.mode == "math", enable_mem_efficient=self.mode == "mem_efficient"):
				attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=dropout_rate)

		attn_weights = None

		attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
		attn_output = self.o_proj(attn_output)

		return attn_output, attn_weights, past_key_value
	"""