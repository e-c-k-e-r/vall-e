# https://github.com/syncdoth/RetNet/
from ....ext.retnet_ts.config import RetNetConfig
from ....ext.retnet_ts.retnet import RetNetModel as RetNetDecoder

# things we're overriding or required to override
from ....ext.retnet_ts.retnet import RetNetDecoderLayer, MultiScaleRetention, theta_shift, RMSNorm, FeedForwardNetwork, get_activation_fn, LayerNorm, RetNetRelPos

import torch
import math
from typing import Dict, List, Optional, Tuple, Union

from torch.utils.checkpoint import checkpoint

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

# removes embed_tokens
def RetNetModel_init(
	self, config, embed_tokens=None, output_projection=None, **kwargs
):
	super(RetNetDecoder, self).__init__(**kwargs)
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

	if (output_projection is None and not config.no_output_layer and config.vocab_size > 0):
		self.output_projection = self.build_output_projection(config)
	else:
		self.output_projection = output_projection

	if config.layernorm_embedding:
		self.layernorm_embedding = LayerNorm(self.embed_dim, eps=config.layernorm_eps) # RMSNorm
	else:
		self.layernorm_embedding = None

	self.layers = torch.nn.ModuleList([])

	for i in range(config.decoder_layers):
		layer = self.build_decoder_layer(
			config,
			depth=i,
		)
		"""
		if config.checkpoint_activations:
			layer = checkpoint_wrapper(layer)
		"""
		self.layers.append(layer)

	self.num_layers = len(self.layers)

	if config.decoder_normalize_before:
		self.layer_norm = LayerNorm(self.embed_dim, eps=config.layernorm_eps) # RMSNorm
	else:
		self.layer_norm = None

	self.retnet_rel_pos = RetNetRelPos(config)
	self.chunkwise_recurrent = config.chunkwise_recurrent
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

RetNetDecoder.__init__ = RetNetModel_init

# restores bias in our FFNs
def RetNetDecoderLayer_init(
	self,
	config,
	depth,
	use_bias=True
):
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
		config,
		self.embed_dim,
		config.decoder_value_embed_dim,
		config.decoder_retention_heads,
		use_bias=use_bias
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

def RetNetDecoderLayer_forward(
	self,
	x,
	incremental_state=None,
	chunkwise_recurrent=False,
	retention_rel_pos=None,
):
	residual = x
	if self.normalize_before:
		x = self.retention_layer_norm(x)

	if x.requires_grad and self.config.checkpoint_activations:
		x = checkpoint(
			self.retention,
			x,
			use_reentrant=False,
			incremental_state=incremental_state,
			rel_pos=retention_rel_pos,
			chunkwise_recurrent=chunkwise_recurrent,
		)
	else:
		x = self.retention(
			x,
			incremental_state=incremental_state,
			rel_pos=retention_rel_pos,
			chunkwise_recurrent=chunkwise_recurrent,
		)
	x = self.dropout_module(x)

	if self.drop_path is not None:
		x = self.drop_path(x)

	x = self.residual_connection(x, residual)
	if not self.normalize_before:
		x = self.retention_layer_norm(x)

	residual = x
	if self.normalize_before:
		x = self.final_layer_norm(x)

	x = self.ffn(x)

	if self.drop_path is not None:
		x = self.drop_path(x)

	x = self.residual_connection(x, residual)
	if not self.normalize_before:
		x = self.final_layer_norm(x)

	return x

RetNetDecoderLayer.__init__ = RetNetDecoderLayer_init
RetNetDecoderLayer.forward = RetNetDecoderLayer_forward
# fixes backwards when using te's autocast
def MultiScaleRetention_init(
	self,
	config,
	embed_dim,
	value_dim,
	num_heads,
	gate_fn="swish",
	use_bias=True,
):
	super(MultiScaleRetention, self).__init__()
	self.config = config
	self.embed_dim = embed_dim
	self.value_dim = value_dim
	self.num_heads = num_heads
	self.head_dim = self.value_dim // num_heads
	self.key_dim = self.embed_dim // num_heads
	self.scaling = self.key_dim**-0.5

	self.gate_fn = get_activation_fn(activation=str(gate_fn))

	self.q_proj = torch.nn.Linear(embed_dim, embed_dim, bias=use_bias)
	self.k_proj = torch.nn.Linear(embed_dim, embed_dim, bias=use_bias)
	self.v_proj = torch.nn.Linear(embed_dim, value_dim, bias=use_bias)
	self.g_proj = torch.nn.Linear(embed_dim, value_dim, bias=use_bias)

	self.out_proj = torch.nn.Linear(value_dim, embed_dim, bias=use_bias)

	self.group_norm = RMSNorm(self.head_dim, eps=config.layernorm_eps, elementwise_affine=False)
	self.reset_parameters()

def MultiScaleRetention_forward(
	self, x, rel_pos, chunkwise_recurrent=False, incremental_state=None
) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor]]:
	bsz, tgt_len, _ = x.size()
	(sin, cos), inner_mask = rel_pos

	q = self.q_proj(x)
	k = self.k_proj(x) * self.scaling
	v = self.v_proj(x)
	g = self.g_proj(x)

	q = q.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
	k = k.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)

	qr = theta_shift(q, sin, cos)
	kr = theta_shift(k, sin, cos)

	if incremental_state is not None:
		output = self.recurrent_forward(qr, kr, v, inner_mask, incremental_state)
	elif chunkwise_recurrent:
		output = self.chunk_recurrent_forward(qr, kr, v, inner_mask)
	else:
		output = self.parallel_forward(qr, kr, v, inner_mask)

	output = self.group_norm(output).reshape(bsz, tgt_len, self.head_dim * self.num_heads)

	output = self.gate_fn(g) * output

	output = self.out_proj(output)

	return output

MultiScaleRetention.__init__ = MultiScaleRetention_init
MultiScaleRetention.forward = MultiScaleRetention_forward