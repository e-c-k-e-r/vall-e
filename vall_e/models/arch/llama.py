# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

import math
import torch
import logging
import random

from typing import Literal, overload, Optional, Tuple, Union, List

from torch import Tensor, nn
from transformers.cache_utils import Cache

from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding, apply_rotary_pos_emb, repeat_kv
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

_logger = logging.getLogger(__name__)

AVAILABLE_ATTENTIONS = []

LN_2 = 0.69314718056

try:
	from sageattention import sageattn
	
	AVAILABLE_ATTENTIONS.append("sageattn")
except Exception as e:
	_logger.warning(f"Error while querying for `sageattn` support: {str(e)}")

try:
	from torch.nn.attention.flex_attention import flex_attention, create_block_mask

	AVAILABLE_ATTENTIONS.append("flex")
except Exception as e:
	_logger.warning(f"Error while querying for `flexattention` support: {str(e)}")

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

class LlamaDecoderLayer_Adapted(LlamaDecoderLayer):
	# apply timestep embedding with attention norm
	# I don't have a concrete idea on how helpful this is, as:
	# * F5-TTS's UNetT implementation doesn't do this
	# * F5-TTS's DiT does this, but only for pre-attention normalization
	# * MaskGCT does this for both
	# * Muse doesn't do this, but instead appends the timestep embedding
	def weigh_by_timestep(
		self,
		hidden_states,
		timesteps,
	):
		if timesteps is None:
			return hidden_states

		for i, timestep in enumerate( timesteps ):
			# invalid
			if not isinstance( timestep, torch.Tensor ):
				continue
			hidden_states[i] *= timestep
		
		return hidden_states

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
		timesteps: Optional[list] = None,
		**kwargs,
	) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
		"""
		Args:
			hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
			attention_mask (`torch.FloatTensor`, *optional*):
				attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
				query_sequence_length, key_sequence_length)` if default attention is used.
			output_attentions (`bool`, *optional*):
				Whether or not to return the attentions tensors of all attention layers. See `attentions` under
				returned tensors for more detail.
			use_cache (`bool`, *optional*):
				If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
				(see `past_key_values`).
			past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
			cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
				Indices depicting the position of the input sequence tokens in the sequence
			position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
				Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
				with `head_dim` being the embedding dimension of each attention head.
			kwargs (`dict`, *optional*):
				Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
				into the model
		"""
		residual = hidden_states

		hidden_states = self.input_layernorm(hidden_states)
		hidden_states = self.weigh_by_timestep( hidden_states, timesteps )
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
		hidden_states = self.weigh_by_timestep( hidden_states, timesteps )

		hidden_states = self.mlp(hidden_states)
		hidden_states = residual + hidden_states

		outputs = (hidden_states,)

		if output_attentions:
			outputs += (self_attn_weights,)

		if use_cache:
			outputs += (present_key_value,)

		return outputs

class LlamaModel_Adapted(LlamaModel):
	def __init__(self, config, *args, **kwargs):
		self.layer_dropout_p = kwargs.pop("layer_dropout_p", 0.1)
		self.early_exit_scale = kwargs.pop("early_exit_scale", 0.1)
		self.early_exit_r = kwargs.pop("early_exit_r", 2)

		#super().__init__(*args, **kwargs)
		super(LlamaModel, self).__init__(config)

		self.padding_idx = config.pad_token_id
		self.vocab_size = config.vocab_size
		self.layers_n = config.num_hidden_layers

		# self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

		self.layers = nn.ModuleList(
			[LlamaDecoderLayer_Adapted(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
		)
		self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
		self.rotary_emb = LlamaRotaryEmbedding(config=config)
		self.gradient_checkpointing = False

		# Initialize weights and apply final processing
		self.post_init()

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
		past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		cache_position: Optional[torch.LongTensor] = None,
		
		layer_skip_lambda = None,
		timesteps: Optional[list] = None,
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

		"""
		if inputs_embeds is None:
			inputs_embeds = self.embed_tokens(input_ids)
		"""

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
					timesteps,
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
					timesteps=timesteps,
				)

			if not self.dropoff_layer( l ):
				hidden_states = layer_outputs[0]

				if use_cache:
					next_decoder_cache = layer_outputs[2 if output_attentions else 1]

			if output_attentions:
				all_self_attns += (layer_outputs[1],)

			# check if we should early-exit
			if layer_skip_lambda and layer_skip_lambda( l, hidden_states ):
				#_logger.info(f"Early exit at layer: {l}")
				break

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