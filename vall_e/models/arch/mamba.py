
from transformers.models.mamba.modeling_mamba import MambaModel
from transformers.models.mamba2.modeling_mamba2 import Mamba2Model

from transformers.models.mamba.configuration_mamba import MambaConfig
from transformers.models.mamba2.configuration_mamba2 import Mamba2Config

"""
# https://github.com/state-spaces/mamba
from torch.utils.checkpoint import checkpoint

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaConfig, MixerModel as MambaMixelModel, layer_norm_fn as MambaLayerNormFn, RMSNorm as MambaRMSNorm

def MambaMixelModel_forward(self, input_ids=None, hidden_states=None, inference_params=None, **mixer_kwargs):
	if hidden_states is None:
		hidden_states = self.embedding(input_ids)
	residual = None
	for layer in self.layers:
		if self.gradient_checkpointing and hidden_states.requires_grad:
			hidden_states, residual = checkpoint( layer, hidden_states, residual, inference_params=inference_params, **mixer_kwargs, use_reentrant=False )
		else:
			hidden_states, residual = layer( hidden_states, residual, inference_params=inference_params, **mixer_kwargs )
	if not self.fused_add_norm:
		residual = (hidden_states + residual) if residual is not None else hidden_states
		hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
	else:
		# Set prenorm=False here since we don't need the residual
		hidden_states = MambaLayerNormFn(
			hidden_states,
			self.norm_f.weight,
			self.norm_f.bias,
			eps=self.norm_f.eps,
			residual=residual,
			prenorm=False,
			residual_in_fp32=self.residual_in_fp32,
			is_rms_norm=isinstance(self.norm_f, MambaRMSNorm)
		)
	return hidden_states

MambaMixelModel.forward = MambaMixelModel_forward
"""