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