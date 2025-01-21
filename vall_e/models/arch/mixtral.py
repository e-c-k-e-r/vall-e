# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py

import math
import torch
import torch.nn.functional as F
from typing import Literal, overload, Optional, Tuple, List, Union

from transformers import MixtralModel, MixtralConfig
from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func, MixtralSparseMoeBlock, MixtralAttention, MixtralDecoderLayer, MixtralRMSNorm, repeat_kv
from transformers.modeling_outputs import BaseModelOutputWithPast, MoeModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.processing_utils import Unpack

from .attention import *

def rotate_half(x):
	"""Rotates half the hidden dims of the input."""
	x1 = x[..., : x.shape[-1] // 2]
	x2 = x[..., x.shape[-1] // 2 :]
	return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
	"""Applies Rotary Position Embedding to the query and key tensors.

	Args:
		q (`torch.Tensor`): The query tensor.
		k (`torch.Tensor`): The key tensor.
		cos (`torch.Tensor`): The cosine part of the rotary embedding.
		sin (`torch.Tensor`): The sine part of the rotary embedding.
		position_ids (`torch.Tensor`):
			The position indices of the tokens corresponding to the query and key tensors. For example, this can be
			used to pass offsetted position ids when working with a KV-cache.
		unsqueeze_dim (`int`, *optional*, defaults to 1):
			The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
			sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
			that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
			k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
			cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
			the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
	Returns:
		`tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
	"""
	cos = cos[position_ids].unsqueeze(unsqueeze_dim)
	sin = sin[position_ids].unsqueeze(unsqueeze_dim)

	q_embed = (q * cos) + (rotate_half(q) * sin)
	k_embed = (k * cos) + (rotate_half(k) * sin)
	return q_embed, k_embed

# This is required because batch sizes > 1 throws errors
def MixtralSparseMoeBlock_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
	""" """
	batch_size, sequence_length, hidden_dim = hidden_states.shape
	if self.training and self.jitter_noise > 0:
		hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
	#hidden_states = hidden_states.view(-1, hidden_dim)
	hidden_states = hidden_states.reshape(-1, hidden_dim)
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

	# One hot encode the selected experts to create an expert mask
	# this will be used to easily index which expert is going to be sollicitated
	expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

	# Loop over all available experts in the model and perform the computation on each expert
	for expert_idx in range(self.num_experts):
		expert_layer = self.experts[expert_idx]
		idx, top_x = torch.where(expert_mask[expert_idx])

		# Index the correct hidden states and compute the expert hidden state for
		# the current expert. We need to make sure to multiply the output hidden
		# states by `routing_weights` on the corresponding tokens (top-1 and top-2)
		current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
		current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

		# However `index_add_` only support torch tensors for indexing so we'll use
		# the `top_x` tensor here.
		final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
	final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
	return final_hidden_states, router_logits

MixtralSparseMoeBlock.forward = MixtralSparseMoeBlock_forward

class MixtralAttention_Adapted(MixtralAttention):
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
		mode = "default" if output_attentions else self.mode
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

		kv_seq_len = key_states.shape[-2]
		if past_key_value is not None:
		    if self.layer_idx is None:
		        raise ValueError(
		            f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
		            "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
		            "with a layer index."
		        )
		    kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
		cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
		query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

		if past_key_value is not None:
		    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
		    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

		# repeat k/v heads if n_kv_heads < n_heads
		key_states = repeat_kv(key_states, self.num_key_value_groups)
		value_states = repeat_kv(value_states, self.num_key_value_groups)

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
			attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
			attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
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
			with torch.nn.attention.sdpa_kernel(self.mode):
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

class MixtralDecoderLayer_Adapted(MixtralDecoderLayer):
	def forward(
		self,
		hidden_states: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
		is_causal: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_value: Optional[Tuple[torch.Tensor]] = None,
		output_attentions: Optional[bool] = False,
		output_router_logits: Optional[bool] = False,
		use_cache: Optional[bool] = False,
		cache_position: Optional[torch.LongTensor] = None,
		position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
		**kwargs: Unpack[FlashAttentionKwargs],
	) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
		"""
		Args:
			hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
			attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
				`(batch, sequence_length)` where padding elements are indicated by 0.
			past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
			output_attentions (`bool`, *optional*):
				Whether or not to return the attentions tensors of all attention layers. See `attentions` under
				returned tensors for more detail.
			output_router_logits (`bool`, *optional*):
				Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
				should not be returned during inference.
			use_cache (`bool`, *optional*):
				If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
				(see `past_key_values`).
			cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
				Indices depicting the position of the input sequence tokens in the sequence.
			kwargs (`dict`, *optional*):
				Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
				into the model
		"""

		residual = hidden_states

		hidden_states = self.input_layernorm(hidden_states)

		# Self Attention
		hidden_states, self_attn_weights, present_key_value = self.self_attn(
			hidden_states=hidden_states,
			position_embeddings=position_embeddings,
			attention_mask=attention_mask,
			is_causal=is_causal,
			position_ids=position_ids,
			past_key_value=past_key_value,
			output_attentions=output_attentions,
			use_cache=use_cache,
			cache_position=cache_position,
			**kwargs,
		)
		hidden_states = residual + hidden_states

		# Fully Connected
		residual = hidden_states
		hidden_states = self.post_attention_layernorm(hidden_states)
		hidden_states, router_logits = self.block_sparse_moe(hidden_states)
		hidden_states = residual + hidden_states

		outputs = (hidden_states,)

		if output_attentions:
			outputs += (self_attn_weights,)

		if output_router_logits:
			outputs += (router_logits,)

		return outputs

class MixtralRotaryEmbedding(torch.nn.Module):
	def __init__(self, config: MixtralConfig, device=None):
		super().__init__()
		# BC: "rope_type" was originally "type"
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
		"""
		dynamic RoPE layers should recompute `inv_freq` in the following situations:
		1 - growing beyond the cached sequence length (allow scaling)
		2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
		"""
		seq_len = torch.max(position_ids) + 1
		if seq_len > self.max_seq_len_cached:  # growth
			inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
			self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
			self.max_seq_len_cached = seq_len

		if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
			# This .to() is needed if the model has been moved to a device after being initialized (because
			# the buffer is automatically moved, but not the original copy)
			self.original_inv_freq = self.original_inv_freq.to(device)
			self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
			self.max_seq_len_cached = self.original_max_seq_len

	@torch.no_grad()
	def forward(self, x, position_ids):
		if "dynamic" in self.rope_type:
			self._dynamic_frequency_update(position_ids, device=x.device)

		# Core RoPE block
		inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
		position_ids_expanded = position_ids[:, None, :].float()
		# Force float32 (see https://github.com/huggingface/transformers/pull/29285)
		device_type = x.device.type
		device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
		with torch.autocast(device_type=device_type, enabled=False):
			freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
			emb = torch.cat((freqs, freqs), dim=-1)
			cos = emb.cos()
			sin = emb.sin()

		# Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
		cos = cos * self.attention_scaling
		sin = sin * self.attention_scaling

		return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class MixtralModel_Adapted(MixtralModel):
	def __init__(self, config: MixtralConfig):
		#super().__init__(config)
		super(MixtralModel, self).__init__(config)

		self.padding_idx = config.pad_token_id
		self.vocab_size = config.vocab_size

		self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
		self.layers = torch.nn.ModuleList(
			[MixtralDecoderLayer_Adapted(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
		)
		self.norm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
		self.rotary_emb = MixtralRotaryEmbedding(config)
		self.gradient_checkpointing = False

		# Initialize weights and apply final processing
		self.post_init()

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

	# gut out the things that just shoves responsibility on SDPA's is_causal generating a mask because this causes problems
	def _update_causal_mask(
		self,
		attention_mask: torch.Tensor,
		input_tensor: torch.Tensor,
		cache_position: torch.Tensor,
		past_key_values: Cache,
		output_attentions: bool,
	):
		"""
		if self.config._attn_implementation == "flash_attention_2":
			if attention_mask is not None and 0.0 in attention_mask:
				return attention_mask
			return None
		"""

		past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
		using_static_cache = isinstance(past_key_values, StaticCache)

		"""
		# For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
		# order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
		# to infer the attention mask.
		# When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
		if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
			if AttentionMaskConverter._ignore_causal_mask_sdpa(
				attention_mask,
				inputs_embeds=input_tensor,
				past_key_values_length=past_seen_tokens,
				is_training=self.training,
			):
				return None
		"""

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
			config=self.config,
			past_key_values=past_key_values,
		)

		if (
			self.config._attn_implementation == "sdpa"
			and attention_mask is not None
			and attention_mask.device.type == "cuda"
			and not output_attentions
		):
			# Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
			# using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
			# Details: https://github.com/pytorch/pytorch/issues/110213
			min_dtype = torch.finfo(dtype).min
			causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

		return causal_mask

	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		is_causal: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[List[torch.FloatTensor]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		output_router_logits: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		cache_position: Optional[torch.LongTensor] = None,
		**flash_attn_kwargs: Unpack[FlashAttentionKwargs],
	) -> Union[Tuple, BaseModelOutputWithPast]:
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_router_logits = (
			output_router_logits if output_router_logits is not None else self.config.output_router_logits
		)
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		use_cache = use_cache if use_cache is not None else self.config.use_cache

		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

		if self.gradient_checkpointing and self.training:
			if use_cache:
				use_cache = False

		if use_cache and past_key_values is None:
			past_key_values = DynamicCache()

		if inputs_embeds is None:
			inputs_embeds = self.embed_tokens(input_ids)

		if cache_position is None:
			past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
			cache_position = torch.arange(
				past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
			)
		if position_ids is None:
			position_ids = cache_position.unsqueeze(0)

		#causal_mask = self._update_causal_mask(
		#	attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
		#)
		# because we can attend to both a causal and a non-causal sequence, generate both masks then pick among which to use per batch
		if is_causal is not None:
			"""
			causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
				attention_mask,
				sequence_length=inputs_embeds.shape[1],
				target_length=attention_mask.shape[-1] if attention_mask is not None else inputs_embeds.shape[1],
				dtype=inputs_embeds.dtype,
				device=inputs_embeds.device,
				cache_position=cache_position,
				batch_size=inputs_embeds.shape[0],
			)
			"""
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
		all_router_logits = () if output_router_logits else None

		for decoder_layer in self.layers:
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
					output_router_logits,
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
					output_router_logits=output_router_logits,
					use_cache=use_cache,
					cache_position=cache_position,
					position_embeddings=position_embeddings,
					**flash_attn_kwargs,
				)

			hidden_states = layer_outputs[0]

			if output_attentions:
				all_self_attns += (layer_outputs[1],)

			if output_router_logits:
				all_router_logits += (layer_outputs[-1],)

		hidden_states = self.norm(hidden_states)

		# add hidden states from the last decoder layer
		if output_hidden_states:
			all_hidden_states += (hidden_states,)

		output = MoeModelOutputWithPast(
			last_hidden_state=hidden_states,
			past_key_values=past_key_values,
			hidden_states=all_hidden_states,
			attentions=all_self_attns,
			router_logits=all_router_logits,
		)
		return output if return_dict else output.to_tuple()