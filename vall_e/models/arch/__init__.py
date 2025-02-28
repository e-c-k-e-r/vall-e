AVAILABLE_ARCHES = []
ERROR_ARCHES = {}

try:
	from .llama import Config as LlamaConfig, Model as LlamaModel, Attention as LlamaAttention, AVAILABLE_ATTENTIONS
	AVAILABLE_ARCHES.append("llama")
except Exception as e:
	ERROR_ARCHES["llama"] = e
	AVAILABLE_ATTENTIONS = []
	pass

"""
try:
	from .transformer import SinusoidalEmbedding, Block as TransformerBlock
	AVAILABLE_ARCHES.append("transformer")
except Exception as e:
	ERROR_ARCHES["transformer"] = e
	pass

try:
	from .retnet import RetNetDecoder, RetNetConfig
	AVAILABLE_ARCHES.append("retnet")
except Exception as e:
	ERROR_ARCHES["retnet"] = e
	pass

try:
	from .retnet_syncdoth.retnet_ts import RetNetDecoder as RetNetDecoder_TS, RetNetConfig as RetNetConfig_TS
	AVAILABLE_ARCHES.append("retnet-ts")
except Exception as e:
	ERROR_ARCHES["retnet-ts"] = e
	pass

try:
	from .retnet_syncdoth.retnet_hf import RetNetDecoder as RetNetDecoder_HF, RetNetConfig as RetNetConfig_HF, RetNetForCausalLM
	AVAILABLE_ARCHES.append("retnet-hf")
except Exception as e:
	ERROR_ARCHES["retnet-hf"] = e
	pass

try:
	from .bitnet import BitNetTransformer
	AVAILABLE_ARCHES.append("bitnet")
except Exception as e:
	ERROR_ARCHES["bitnet"] = e
	pass

try:
	from .mixtral import MixtralModel, MixtralConfig, MixtralAttention, MixtralAttention_Adapted, MixtralModel_Adapted, load_balancing_loss_func
	AVAILABLE_ARCHES.append("mixtral")
except Exception as e:
	ERROR_ARCHES["mixtral"] = e

try:
	from .mamba import MambaModel, Mamba2Model, MambaConfig, Mamba2Config
	AVAILABLE_ARCHES.append("mamba")
	AVAILABLE_ARCHES.append("mamba2")
except Exception as e:
	ERROR_ARCHES["mamba"] = e
	ERROR_ARCHES["mamba2"] = e
"""
"""
try:
	from .mamba import MambaMixelModel, MambaLMHeadModel, MambaConfig
	AVAILABLE_ARCHES.append("mamba")
	AVAILABLE_ARCHES.append("mamba2")
except Exception as e:
	ERROR_ARCHES["mamba"] = e
	ERROR_ARCHES["mamba2"] = e

try:
	from .mamba_vasqu import Mamba2Model_HF, Mamba2Config_HF
	AVAILABLE_ARCHES.append("mamba2-hf")
except Exception as e:
	ERROR_ARCHES["mamba2-hf"] = e
"""