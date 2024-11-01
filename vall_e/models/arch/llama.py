# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

import math
import torch
import logging
import random

from typing import Literal, overload, Optional, Tuple, Union, List, Unpack

from torch import Tensor, nn
from transformers.cache_utils import Cache

from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv

_logger = logging.getLogger(__name__)

AVAILABLE_ATTENTIONS = []

LN_2 = 0.69314718056

try:
	from transformers.utils import is_flash_attn_2_available

	if is_flash_attn_2_available():
		AVAILABLE_ATTENTIONS.append("flash_attention_2")
except Exception as e:
	_logger.warning(f"Error while querying for `flash_attention_2` support: {str(e)}")

try:
	from .attention.fused import attention as _fused_attention
	def fused_attn_func(q, k, v, softmax_scale=None, causal=False, *args, **kwargs):
		return _fused_attention( q, k, v, causal, softmax_scale )
	
	AVAILABLE_ATTENTIONS.append("fused_attn")
except Exception as e:
	_logger.warning(f"Error while querying for `fused_attn` support: {str(e)}")


is_rocm = any("AMD" in torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count()))
is_ampere_or_newer_gpu = any(torch.cuda.get_device_properties(i).major >= 8 for i in range(torch.cuda.device_count()))

try:
	if is_rocm:
		# requires pain to set up on Navi3, and for no backwards (training) support
		from flash_attn import flash_attn_func
		AVAILABLE_ATTENTIONS.append("flash_attn")

	elif not is_ampere_or_newer_gpu:
		# Uses https://github.com/ZRayZzz/flash-attention-v100/
		# Currently doesn't work because it's hard-coded to use a head dim of 128, will throw NaNs otherwise...
		from flash_attn_v100 import flash_attn_func as flash_attn_v100_func

		AVAILABLE_ATTENTIONS.append("flash_attn")
		AVAILABLE_ATTENTIONS.append("flash_attn_v100") # needed to signal to use padding
		def flash_attn_func(q, k, v, softmax_scale=None, causal=False, *args, **kwargs):
			return flash_attn_v100_func(
				q,
				k,
				v,
				softmax_scale,
				causal
			)
	else:
		# Borrowed from https://github.com/turboderp/exllamav2/blob/master/exllamav2/attn.py#L32
		# Adapted to provide flash_attn_v1 support
		import flash_attn
		flash_attn_ver = [int(t) for t in flash_attn.__version__.split(".") if t.isdigit()]

		if flash_attn_ver <= [1, 0, 9]:
			AVAILABLE_ATTENTIONS.append("flash_attn")
			from flash_attn.flash_attn_interface import flash_attn_unpadded_func
			from einops import rearrange

			# converts the flash_attn_2 calling convention to flash_attn_1's
			def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, return_attn_probs=False, deterministic=False, *args, **kwargs):
				batch_size, seqlen_q = q.shape[0], q.shape[1]
				seqlen_k = k.shape[1]
				q, k, v = [rearrange(x, 'b s ... -> (b s) ...').contiguous() for x in [q, k, v]]

				cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q.device)
				cu_seqlens_k = cu_seqlens_q

				return flash_attn_unpadded_func(
					q, k, v,
					cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
					dropout_p, softmax_scale, causal, return_attn_probs, deterministic
				)
			
			has_flash_attn = True
		elif [2, 2, 1] <= flash_attn_ver < [2, 5, 7]:
			AVAILABLE_ATTENTIONS.append("flash_attn")
			from flash_attn import flash_attn_func
			has_flash_attn = True
		elif [2, 5, 7] <= flash_attn_ver:
			AVAILABLE_ATTENTIONS.append("flash_attn")
			from flash_attn import flash_attn_func, flash_attn_with_kvcache

			signature = list(inspect.signature(flash_attn_func).parameters)
			has_flash_attn_with_window = "window_size" in signature
			has_flash_attn_with_softcap = "softcap" in signature

			import flash_attn_2_cuda as flash_attn_cuda

			has_flash_attn = True
			has_flash_attn_with_paged = True
except Exception as e:
	_logger.warning(f"Error while querying for `flash_attn` support: {str(e)}")

try:
	from xformers.ops.fmha import memory_efficient_attention
	from xformers.ops.fmha.attn_bias import LowerTriangularFromBottomRightMask, LowerTriangularMask

	AVAILABLE_ATTENTIONS.append("xformers")
except Exception as e:
	_logger.warning(f"Error while importing `xformers`: {str(e)}")

# to-do: find a better way to query for if there's available kernels since these return true regardless
if torch.backends.cuda.flash_sdp_enabled():
	AVAILABLE_ATTENTIONS.append("flash_(sdpa)")

if torch.backends.cuda.mem_efficient_sdp_enabled():
	AVAILABLE_ATTENTIONS.append("mem_efficient")	

if torch.backends.cuda.math_sdp_enabled():
	AVAILABLE_ATTENTIONS.append("math")	

if torch.backends.cuda.cudnn_sdp_enabled():
	AVAILABLE_ATTENTIONS.append("cudnn")	

if AVAILABLE_ATTENTIONS:
	AVAILABLE_ATTENTIONS.append("sdpa")
	AVAILABLE_ATTENTIONS.append("default")

class LlamaAttention_Adapted(LlamaAttention):
	def __init__(self, *args, **kwargs):
		self.mode = kwargs.pop("mode", "sdpa")

		if self.mode == "math":
			self.mode = torch.nn.attention.SDPBackend.MATH
		elif self.mode == "mem_efficient":
			self.mode = torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
		elif self.mode == "flash_(sdpa)":
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
		mode = "default" if output_attentions else self.mode
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
			# TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
			# to be able to avoid many of these transpose/reshape/view.
			query_states = query_states.transpose(1, 2)
			key_states = key_states.transpose(1, 2)
			value_states = value_states.transpose(1, 2)

			"""
			# In PEFT, usually we cast the layer norms in float32 for training stability reasons
			# therefore the input hidden states gets silently casted in float32. Hence, we need
			# cast them back in the correct dtype just to be sure everything works as expected.
			# This might slowdown training & inference so it is recommended to not cast the LayerNorms
			# in fp32. (LlamaRMSNorm handles it correctly)

			input_dtype = query_states.dtype
			if input_dtype == torch.float32:
				if torch.is_autocast_enabled():
					target_dtype = torch.get_autocast_gpu_dtype()
				# Handle the case where the model is quantized
				elif hasattr(self.config, "_pre_quantization_dtype"):
					target_dtype = self.config._pre_quantization_dtype
				else:
					target_dtype = self.q_proj.weight.dtype

				query_states = query_states.to(target_dtype)
				key_states = key_states.to(target_dtype)
				value_states = value_states.to(target_dtype)
			"""

			if mode == "flash_attn":
				attn_output = flash_attn_func(
					query_states,
					key_states,
					value_states,
					causal=True,
					softmax_scale=1.0 / math.sqrt(self.head_dim),
					dropout_p=dropout_rate,
				)
				
				attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
			elif mode == "xformers":
				attn_output = memory_efficient_attention(
					query_states,
					key_states,
					value_states,
					attn_bias = LowerTriangularMask() if attention_mask is None or attention_mask[0, 0, 0, 1] == 0 else None,
					scale = 1.0 / math.sqrt(self.head_dim),
					p=dropout_rate
				)
				attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

			attn_output = self.o_proj(attn_output)
			return attn_output, attn_scores, past_key_value

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
		
		if mode in ["fused_attn"]:
			attn_output = fused_attn_func(
				query_states,
				key_states,
				value_states,
				causal=True,
				softmax_scale=1.0 / math.sqrt(self.head_dim),
				dropout_p=dropout_rate,
			)
		elif mode in ["default"]:
			attn_scores = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
			# cringe logic
			attn_weights = (attn_scores + causal_mask) if attention_mask is not None else (attn_scores)
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
			with torch.nn.attention.sdpa_kernel(self.mode):
				attn_output = torch.nn.functional.scaled_dot_product_attention(
					query_states,
					key_states,
					value_states,
					attn_mask=causal_mask,
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

class LlamaModel_Adapted(LlamaModel):
	def __init__(self, *args, **kwargs):
		self.layer_dropout_p = kwargs.pop("layer_dropout_p", 0.1)
		self.early_exit_scale = kwargs.pop("early_exit_scale", 0.1)
		self.early_exit_r = kwargs.pop("early_exit_r", 2)

		super().__init__(*args, **kwargs)

		self.layers_n = len(self.layers)
	def dropoff_layer( self, l ):
		if not self.training:
			return False

		# this could probably a LUT but I'm not fiending for aggressive mal-optimizations
		D = math.exp((l * LN_2) / (self.layers_n - 1)) - 1
		P = D * self.layer_dropout_p
		return random.random() < P

	def cirriculum( self, l, t=None ):
		# no timestep data passed, just treat all layers as enabled
		# there doesn't seem /too/ bad of a performance hit, but the paper mentions it affecting accuracy of the last layer if all layers had early exit
		if t is None:
			return 1
		
		# YUCK
		# this guarantees at least R layers are active at all intervals, which is important because this gives a division by zero otherwise
		for i in range(self.early_exit_r):
			if l == ((t % self.layers_n) + i * (self.layers_n // self.early_exit_r)) % self.layers_n:
				return 1
		return 0

	def early_exit_loss( self, losses, t=None ):
		return sum([ self.normalized_per_layer_loss_scale( l, t ) * losses[l] for l in range(0, self.layers_n) ])

	def normalized_per_layer_loss_scale( self, l, t=None ):
		return (self.cirriculum(l, t) * self.early_exit_factor( l )) / sum([ self.cirriculum(i, t) * self.early_exit_factor( i ) for i in range(0, self.layers_n) ])

	def early_exit_factor( self, l ):
		if 0 <= l and l < self.layers_n:
			return self.early_exit_scale * sum([ i for i in range(0, l) ])
		return self.layers_n - 1 + self.early_exit_scale * sum([ i for i in range(0, self.layers_n - 1) ])

	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		cache_position: Optional[torch.LongTensor] = None,
		early_exit_layer: Optional[int] = -1,
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

		if inputs_embeds is None:
			inputs_embeds = self.embed_tokens(input_ids)

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

		causal_mask = self._update_causal_mask(
			attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
		)
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
					causal_mask,
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
					attention_mask=causal_mask,
					position_ids=position_ids,
					past_key_value=past_key_values,
					output_attentions=output_attentions,
					use_cache=use_cache,
					cache_position=cache_position,
					position_embeddings=position_embeddings,
				)

			if not self.dropoff_layer( l ):
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