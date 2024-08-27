# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py

import math
import torch
import torch.nn.functional as F
from typing import Literal, overload, Optional, Tuple
from transformers.cache_utils import Cache

from transformers import MixtralModel, MixtralConfig
from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func, MixtralSparseMoeBlock, MixtralAttention, apply_rotary_pos_emb, repeat_kv

try:
	from .llama import flash_attn_func
except Exception as e:
	pass

try:
	from .llama import fused_attn_func
except Exception as e:
	pass

try:
	from .llama import memory_efficient_attention
except Exception as e:
	pass

# This is required because batch sizes > 1 throws errors
def MixtralSparseMoeBlock_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
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

MixtralSparseMoeBlock.forward = MixtralSparseMoeBlock_forward

class MixtralAttention_Adapted(MixtralAttention):
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

	# Adapted from MixtralAttention.forward
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
	) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
		if output_attentions:
			# TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
			"""
			logger.warning_once(
				"MixtralModel is using MixtralSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
				'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
			)
			"""
			return super().forward(
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				past_key_value=past_key_value,
				output_attentions=output_attentions,
				use_cache=use_cache,
			)

		dropout_rate = self.attention_dropout if self.training else 0.0
		bsz, q_len, _ = hidden_states.size()

		query_states = self.q_proj(hidden_states)
		key_states = self.k_proj(hidden_states)
		value_states = self.v_proj(hidden_states)

		query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
		key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
		value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

		kv_seq_len = key_states.shape[-2]
		if past_key_value is not None:
			kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
		
		if position_embeddings is None:
			cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
		else:
			cos, sin = position_embeddings

		query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

		if past_key_value is not None:
			cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
			key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

		key_states = repeat_kv(key_states, self.num_key_value_groups)
		value_states = repeat_kv(value_states, self.num_key_value_groups)

		if self.mode in ["xformers", "flash_attn"]:
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

			if self.mode == "flash_attn":
				attn_output = flash_attn_func(
					query_states,
					key_states,
					value_states,
					causal=True,
					softmax_scale=1.0 / math.sqrt(self.head_dim),
					dropout_p=dropout_rate,
				)
				
				attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
			elif self.mode == "xformers":
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
			return attn_output, None, past_key_value

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
		
		if self.mode in ["fused_attn"]:
			attn_output = fused_attn_func(
				query_states,
				key_states,
				value_states,
				causal=True,
				softmax_scale=1.0 / math.sqrt(self.head_dim),
				dropout_p=dropout_rate,
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

		attn_output = attn_output.transpose(1, 2).contiguous()
		attn_output = attn_output.view(bsz, q_len, -1)

		attn_output = self.o_proj(attn_output)

		return attn_output, None, past_key_value