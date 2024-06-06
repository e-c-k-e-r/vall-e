AVAILABLE_ARCHES = []

try:
	from .transformer import SinusoidalEmbedding, Block as TransformerBlock
	AVAILABLE_ARCHES.append("transformer")
except Exception as e:
	print("Error importing `transformer` arch:", e)
	pass

try:
	from .retnet import RetNetDecoder, RetNetConfig
	AVAILABLE_ARCHES.append("retnet")
except Exception as e:
	print("Error importing `retnet` arch:", e)
	pass

try:
	from .retnet_syncdoth.retnet_ts import RetNetDecoder as RetNetDecoder_TS, RetNetConfig as RetNetConfig_TS
	AVAILABLE_ARCHES.append("retnet-ts")
except Exception as e:
	print("Error importing `retnet-ts` arch:", e)
	pass

try:
	from .retnet_syncdoth.retnet_hf import RetNetDecoder as RetNetDecoder_HF, RetNetConfig as RetNetConfig_HF, RetNetForCausalLM
	AVAILABLE_ARCHES.append("retnet-hf")
except Exception as e:
	print("Error importing `retnet-hf` arch:", e)
	pass

try:
	from .llama import LlamaModel, LlamaConfig, AVAILABLE_ATTENTIONS, LlamaAttention, LlamaAttention_Base, LlamaForCausalLM
	AVAILABLE_ARCHES.append("llama")
except Exception as e:
	print("Error importing `llama` arch:", e)
	pass

try:
	from .bitnet import BitNetTransformer
	AVAILABLE_ARCHES.append("bitnet")
except Exception as e:
	print("Error importing `bitnet` arch:", e)
	pass

try:
	from .mixtral import MixtralModel, MixtralConfig
	AVAILABLE_ARCHES.append("mixtral")
except Exception as e:
	print("Error importing `mixtral` arch:", e)

try:
	from .mamba import MambaMixelModel, MambaLMHeadModel
	AVAILABLE_ARCHES.append("mamba")
	AVAILABLE_ARCHES.append("mamba2")
except Exception as e:
	print("Error importing `mamba` arch:", e)