"""
Contains the zoo of alternative attention mechanisms
To-do: align it better with nu-modelling_llama.py's attention selection mechanism
"""

import logging
import torch

_logger = logging.getLogger(__name__)

AVAILABLE_ATTENTIONS = []

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
	from .fused import attention as _fused_attention
	def fused_attn_func(q, k, v, softmax_scale=None, causal=False, *args, **kwargs):
		return _fused_attention( q, k, v, causal, softmax_scale )
	
	AVAILABLE_ATTENTIONS.append("fused_attn")
except Exception as e:
	_logger.warning(f"Error while querying for `fused_attn` support: {str(e)}")

# https://github.com/lucidrains/native-sparse-attention-pytorch/
try:
	from native_sparse_attention_pytorch.native_sparse_attention import SparseAttention
	from native_sparse_attention_pytorch.compress_networks import GroupedMLP

	# patiently waiting for specifying attention masks both for padded sequences and non-causal ones
	# it might not be necessary since ar+nar-len-llama-8 was able to be "repaired" from the NAR being trained with a causal mask initially
	class NativeSparseAttention(SparseAttention):
		def __init__(self, config, layer_idx):
			dim = config.hidden_size
			heads = config.num_attention_heads
			dim_head = getattr(config, "head_dim", dim // heads)
			kv_heads = config.num_key_value_heads
			causal = False # config.causal # to-do: handle split-causal attention like I do for normal attention
			# for now though leave it as false since the mask transformer variant of VALL-E is much more preferable to the causal variant

			# to-do: figure out these settings best for VALL-E
			compress_block_size = 16
			sliding_window_size = 64 # really don't want sliding attention due to the nature of the sequence
			selection_block_size = 16
			num_selected_blocks = 4
			num_compressed_mem_kv = 1
			
			compress_mlp = GroupedMLP(
				dim_head = dim_head,
				compress_block_size = compress_block_size,
				heads = heads,
			)
			
			self.config = config
			self.layer_idx = layer_idx

			super().__init__(
				dim = dim,
				dim_head = dim_head,
				heads = heads,
				kv_heads = kv_heads,

				sliding_window_size = sliding_window_size,
				compress_block_size = compress_block_size,
				selection_block_size = selection_block_size,
				num_selected_blocks = num_selected_blocks,
				num_compressed_mem_kv = num_compressed_mem_kv,

				causal = causal,

				norm = False, # pre/post norm is done here already
				use_diff_topk = True,
				use_triton_kernel = False,
				interpolated_importance_score = False,
				query_heads_share_selected_kv = True, # if set to True, importance score is averaged across query heads to select top-n buckets of kv per kv head - but can be set to False for each query head within a group to look at different sets of kv buckets. will be more memory and compute of course
				compress_mlp = compress_mlp,
				compress_mlp_expand_factor = 4.,
				strategy_combine_mlp = None
			)

	AVAILABLE_ATTENTIONS.append("sparse")
except Exception as e:
	_logger.warning(f"Error while querying for `SparseAttention` support: {str(e)}")
	pass

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