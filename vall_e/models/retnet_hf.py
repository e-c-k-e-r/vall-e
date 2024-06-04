# https://github.com/syncdoth/RetNet/
from ..ext.retnet_hf.configuration_retnet import RetNetConfig
from ..ext.retnet_hf.modeling_retnet import RetNetModel as RetNetDecoder, RetNetForCausalLM

# things we're overriding or required to override
from ..ext.retnet_hf.modeling_retnet import RetNetDecoderLayer, MultiScaleRetention, theta_shift, split_heads, RMSNorm, FeedForwardNetwork, get_activation_fn, LayerNorm, RetNetRelPos

import torch
import math
from typing import Dict, List, Optional, Tuple, Union

# required to have compatibile LayerNorm
def FeedForwardNetwork_init(
	self,
	embed_dim,
	ffn_dim,
	activation_fn,
	dropout,
	activation_dropout,
	layernorm_eps,
	subln=True,
	use_rms_norm=False,
):
	super(FeedForwardNetwork, self).__init__()
	self.embed_dim = embed_dim
	self.activation_fn = get_activation_fn(activation=str(activation_fn))
	self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
	self.dropout_module = torch.nn.Dropout(dropout)
	self.fc1 = torch.nn.Linear(self.embed_dim, ffn_dim)
	self.fc2 = torch.nn.Linear(ffn_dim, self.embed_dim)
	self.ffn_layernorm = LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None

FeedForwardNetwork.__init__ = FeedForwardNetwork_init

def RetNetModel_init(
	self,
	config: RetNetConfig,
	embed_tokens: torch.nn.Embedding = None,
	tensor_parallel: bool = False,
):
	super(RetNetDecoder, self).__init__(config)
	self.config = config

	self.dropout_module = torch.nn.Dropout(config.dropout)

	self.embed_dim = config.decoder_embed_dim
	self.embed_scale = (
		1.0 if config.no_scale_embedding else math.sqrt(self.embed_dim)
	)

	if embed_tokens is None and config.vocab_size:
		embed_tokens = torch.nn.Embedding(
			config.vocab_size, config.decoder_embed_dim, config.pad_token_id
		)
	self.embed_tokens = embed_tokens

	if config.layernorm_embedding:
		self.layernorm_embedding = LayerNorm(self.embed_dim, eps=config.layernorm_eps) # RMSNorm
	else:
		self.layernorm_embedding = None

	self.layers = torch.nn.ModuleList([])

	for i in range(config.decoder_layers):
		self.layers.append(
			RetNetDecoderLayer(config, depth=i, tensor_parallel=tensor_parallel)
		)

	self.decoder_layers = len(self.layers)

	if config.decoder_normalize_before:
		self.layer_norm = LayerNorm(self.embed_dim, eps=config.layernorm_eps) # RMSNorm
	else:
		self.layer_norm = None

	self.retnet_rel_pos = RetNetRelPos(config)
	self.recurrent_chunk_size = config.recurrent_chunk_size

	if config.deepnorm:
		init_scale = math.pow(8.0 * config.decoder_layers, 0.25)
		for name, p in self.named_parameters():
			if (
				"fc1" in name
				or "fc2" in name
				or "out_proj" in name
				or "v_proj" in name
			):
				p.data.div_(init_scale)

	if config.subln and not config.use_glu:
		init_scale = math.sqrt(math.log(config.decoder_layers * 2))
		for name, p in self.named_parameters():
			if (
				"fc1" in name
				or "fc2" in name
				or "out_proj" in name
				or "v_proj" in name
			):
				p.data.mul_(init_scale)

	self.gradient_checkpointing = True
	self.post_init()

RetNetDecoder.__init__ = RetNetModel_init

# restores bias in our FFNs
def RetNetDecoderLayer_init(self, config: RetNetConfig, depth: int, tensor_parallel: bool = False):
	super(RetNetDecoderLayer, self).__init__()
	self.config = config
	self.embed_dim = config.decoder_embed_dim
	self.dropout_module = torch.nn.Dropout(config.dropout)

	if config.drop_path_rate > 0:
		drop_path_prob = np.linspace(
			0, config.drop_path_rate, config.decoder_layers
		)[depth]
		self.drop_path = DropPath(drop_path_prob)
	else:
		self.drop_path = None

	self.retention = MultiScaleRetention(
		config, use_bias=True, tensor_parallel=tensor_parallel
	)

	self.normalize_before = config.decoder_normalize_before

	self.retention_layer_norm = LayerNorm(self.embed_dim, eps=config.layernorm_eps) # RMSNorm

	self.ffn_dim = config.decoder_ffn_embed_dim

	self.ffn = self.build_ffn()

	self.final_layer_norm = LayerNorm(self.embed_dim, eps=config.layernorm_eps) # RMSNorm

	if config.deepnorm:
		self.alpha = math.pow(2.0 * config.decoder_layers, 0.25)
	else:
		self.alpha = 1.0

RetNetDecoderLayer.__init__ = RetNetDecoderLayer_init
# fixes backwards when using te's autocast
def MultiScaleRetention_forward(
		self,
		hidden_states: torch.Tensor,
		rel_pos: Tuple[Tuple[torch.Tensor]],
		retention_mask: Optional[torch.Tensor] = None,
		past_key_value: Optional[Tuple[torch.Tensor]] = None,
		forward_impl: str = "parallel",
		output_retentions: Optional[bool] = False,
	) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor]]:
		B, T, H = hidden_states.size()
		(sin, cos), decay_mask = rel_pos
		# projections
		q = self.q_proj(hidden_states)
		k = self.k_proj(hidden_states) * self.scaling  # for scaled dot product
		v = self.v_proj(hidden_states)
		g = self.g_proj(hidden_states)
		# multi-head
		q, k, v = split_heads((q, k, v), B, T, self.num_heads)

		# rotate
		# NOTE: theta_shift has bug with mps device.
		qr = theta_shift(q, sin, cos)
		kr = theta_shift(k, sin, cos)

		# retention
		if forward_impl == "parallel":
			retention_out, curr_kv, retention_weights = self.parallel_retention(
				qr, kr, v, decay_mask
			)
		elif forward_impl == "recurrent":
			retention_out, curr_kv = self.recurrent_retention(
				qr,
				kr,
				v,
				decay_mask,
				past_key_value=past_key_value,
				retention_mask=retention_mask,
			)
		elif forward_impl == "chunkwise":
			retention_out, curr_kv = self.chunkwise_retention(qr, kr, v, decay_mask)
		else:
			raise ValueError(f"forward_impl {forward_impl} not supported.")

		# concaat heads
		normed = self.group_norm(retention_out).reshape(B, T, self.value_dim)
		# out gate & proj
		out = self.gate_fn(g) * normed
		out = self.out_proj(out)

		outputs = (out, curr_kv)
		if output_retentions:
			outputs += (retention_weights,) if forward_impl == "parallel" else (None,)
		return outputs

MultiScaleRetention.forward = MultiScaleRetention_forward