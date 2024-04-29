import math
import torch
import torch.nn.functional as F
import traceback
import numpy as np
import re

from typing import Literal, overload
from functools import partial
from einops import rearrange

from torch import Tensor, einsum, nn
from torch.nn import Embedding
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy, MulticlassPrecision

from ..samplers import reptition_penalize, length_penalize, top_k_top_p_filtering, dynamic_temperature, top_k_logits_list, mirostat_sample

try:
	from .transformer import SinusoidalEmbedding, Block as TransformerBlock
except Exception as e:
	print("Error importing `transformer` arch:", e)
	pass

try:
	#from .retnet import RetNetDecoder, RetNetConfig
	from .retnet_ts import RetNetDecoder, RetNetConfig
except Exception as e:
	print("Error importing `retnet` arch:", e)
	pass

from .retnet_hf import RetNetDecoder as RetNetDecoder_HF, RetNetConfig as RetNetConfig_HF
"""
try:
except Exception as e:
	print("Error importing `retnet-hf` arch:", e)
	pass
"""

try:
	from transformers import LlamaModel, LlamaConfig
except Exception as e:
	print("Error importing `llama` arch:", e)
	pass

try:
	from transformers import MistralModel, MistralConfig
except Exception as e:
	print("Error importing `mistral` arch:", e)
	pass

try:
	from bitnet.bit_transformer import Transformer as BitNetTransformerBlock, RMSNorm as BitNetRMSNorm

	# override because bitnet's BitNetTransformer includes an embedding input / classifier output layers inside of it, which isn't favorable
	class BitNetTransformer(nn.Module):
		def __init__(
			self,
			dim: int,
			depth: int,
			num_tokens: int,
			heads=8,
			ff_mult=4,
		):
			super().__init__()

			self.transformer = BitNetTransformerBlock( dim=dim, depth=depth, heads=heads, ff_mult=ff_mult )
			self.norm = BitNetRMSNorm(dim)

		def forward(self, x):
			x = self.transformer(x)
			return self.norm( x )

	"""
	from bitnet import BitNetTransformer
	def NoEmbedding_BitNetTransformer_Forward(self, x):
		x = self.transformer(x)
		return self.to_logits[0](x)

	BitNetTransformer.forward = NoEmbedding_BitNetTransformer_Forward 
	"""

except Exception as e:
	print("Error importing `bitnet` arch:", e)
	pass

try:
	from transformers import MixtralModel, MixtralConfig
	from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func, MixtralSparseMoeBlock

	# This is required because batch sizes > 1 throws errors
	def Fixed_MixtralSparseMoeBlock_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
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

	Original_MixtralSparseMoeBlock_forward = MixtralSparseMoeBlock.forward
	MixtralSparseMoeBlock.forward = Fixed_MixtralSparseMoeBlock_forward

except Exception as e:
	print("Error importing `mixtral` arch:", e)

def _create_mask(l, device):
	"""1 is valid region and 0 is invalid."""
	seq = torch.arange(max(l), device=device).unsqueeze(0)  # (1 t)
	stop = torch.tensor(l, device=device).unsqueeze(1)  # (b 1)
	return (seq < stop).float()  # (b t)

def _join(x: tuple[Tensor], sep: Tensor):
	"""
	Args:
		x: (k t d)
		sep: (d)
	"""
	ret = x[0]
	for i in range(1, len(x)):
		ret = torch.cat((ret, sep[None], x[i]), dim=0)
	return ret

def list_to_tensor(x_list: list[Tensor], pattern="t b c -> b t c"):
	"""
	Args:
		x_list: [(t d)]
	Returns:
		x: (? ? ?)
		m: (? ? ?), same as x
	"""
	l = list(map(len, x_list))
	x = rearrange(pad_sequence(x_list), pattern)
	m = _create_mask(l, x_list[0].device)
	m = m.t().unsqueeze(-1)  # (t b 1)
	m = rearrange(m, pattern)
	m = m.to(x)
	return x, m

# automagically parses a batch-list and returns it as a list
"""
class Embedding(nn.Embedding):
	def forward(self, x_list: list[Tensor]) -> list[Tensor]:
		if len(x_list) == 0:
			return []
		return super().forward(torch.cat(x_list)).split([*map(len, x_list)])
"""

# Deprecated implementation
class MultiEmbedding(nn.Module):
	def __init__(self, max_n_levels, n_tokens, token_dim, monolithic=False):
		super().__init__()
		self.monolithic = monolithic
		self.max_n_levels = max_n_levels
		self.n_tokens = n_tokens
		self.weight = nn.Parameter(torch.randn(max_n_levels, n_tokens, token_dim))

	# to-do: select quant level from given quant_levels tensor if given (i.e. through the resp_emb)
	# I imagine this is an oversight in the NAR.
	def forward(self, x_list: list[Tensor], quant_levels: Tensor | None = None) -> list[Tensor]:
		if len(x_list) == 0:
			return []

		# this "strategy" will reserve the weight[0] for te AR and weight[1:] for the NAR
		# the NAR cannot share RVQ-bin level 0 with the AR for the resp_emb
		if self.monolithic:
			w = self.weight[:1] if quant_levels is None else self.weight[1:]
		else:
			w = self.weight

		padded_x_list = []

		for i, xi in enumerate(x_list):
			xi = F.one_hot(xi.to(torch.int64), num_classes=self.n_tokens)  # t l' k
			wi = w.shape[0] - xi.shape[1]
			xi = F.pad(xi, (0, 0, 0, wi))  # t l k
			padded_x_list.append(xi.to(w))

		x = torch.cat(padded_x_list)  # n l k
		x = einsum("l k d, n l k -> n d", w, x)

		x_list = x.split([*map(len, x_list)])

		return x_list

# Embedding that sums each RVQ-bin level within a given input acoustic prompt
class AudioEmbedding(nn.Module):
	def __init__(
		self,
		l_tokens: int, # list of number of tokens (needed because AR resps includes stop token)
		token_dim: int, # dimensionality of the embedding
		levels: int | None = None, # number of RVQ-bins (I don't remember the specifics)
		sums: bool = True # whether to sum all previous layers of embeddings to factor in other RVQ bin levels (I do not know which way is better)
	):
		super().__init__()
		# array of embeddings
		#   proms are [0, prom_levels]
		#   resp are split to where [0] is for the AR, and [1:] are reserved for NAR
		self.embeddings = nn.ModuleList([nn.Embedding(n_tokens, token_dim) for n_tokens in l_tokens])
		# weight influencer for the influence for each level (desu this should be really useless because the weights in the embedding themselves should factor this)
		self.weight = nn.ParameterList([nn.Parameter( torch.Tensor([1]) ) for i in range(levels)]) if levels is not None else None
		# 
		self.sums = sums
	
	def forward(self, xi: Tensor, quant_levels: Tensor | None = None ) -> Tensor:
		# prom
		if quant_levels is None and xi.shape[-1] > 1:
			if self.sums:
				x = sum( [ self.embeddings[k]( xi[:, k] ) * (self.weight[k] if self.weight is not None else 1) for k in range(xi.shape[-1]) ] )
			else:
				k = 0 # only use the most significant RVQ bin level for the input prom
				x = self.embeddings[k]( xi[:, k] ) * (self.weight[k] if self.weight is not None else 1)
		# AR resp
		elif quant_levels is None or quant_levels == 0:
			x = self.embeddings[0]( xi[:, 0] )
		# NAR resp
		else:
			if self.sums:
				x = sum( [ self.embeddings[k+1]( xi[:, k] ) * (self.weight[k+1] if self.weight is not None else 1) for k in range(xi.shape[-1]) ] )
			else:
				k = xi.shape[-1] - 1 # only use the previous RVQ bin level for the current resp embedding
				x = self.embeddings[k+1]( xi[:, k] ) * (self.weight[k+1] if self.weight is not None else 1)
		
		return x

class Base(nn.Module):
	@property
	def causal(self) -> bool:
		raise NotImplementedError

	@property
	def arch_type(self) -> str:
		raise NotImplementedError

	@property
	def norm_type(self):
		raise NotImplementedError

	@property
	def n_prom_levels(self) -> int:
		raise NotImplementedError

	@property
	def n_resp_levels(self) -> int:
		raise NotImplementedError

	@property
	def n_max_levels(self) -> int:
		raise NotImplementedError
	
	@property
	def n_langs(self) -> int:
		raise NotImplementedError
	
	@property
	def n_tasks(self) -> int:
		raise NotImplementedError

	@property
	def n_tones(self) -> int:
		raise NotImplementedError

	@property
	def recurrent_chunk_size(self) -> int:
		raise NotImplementedError

	@property
	def rotary_embedding_base(self) -> float:
		return 10000
	
	@property
	def interleave(self) -> bool:
		return False

	@property
	def monolithic(self) -> bool:
		return False

	@property
	def version(self) -> int:
		return 1

	@property
	def stop_token(self):
		if not self.causal:
			raise ValueError("Not using stop token!")
		return self.n_tokens

	@property
	def ignore_index(self):
		return -100

	def __init__(
		self,
		n_tokens: int = 1024,
		d_model: int = 512,
		n_heads: int = 8,
		n_layers: int = 12,
		p_dropout: float = 0.1,

		n_experts: int = 1,

		l_padding: int = 0,

		training = True, 
		config = None, 
	):
		super().__init__()
		self.training = training
		self.config = config
		self.activation_checkpointing = self.config.activation_checkpointing if self.config is not None else True

		self.n_tokens = n_tokens
		self.d_model = d_model
		self.n_heads = n_heads
		self.n_layers = n_layers
		self.n_experts = n_experts
		
		self.l_padding = l_padding

		# +1 to include the stop token
		# to-do: undo this dogshit mistake; tasks tokens should be delegated to its own embedding
		n_prom_tokens = n_tokens
		n_resp_tokens = n_tokens + (1 if self.causal else 0) # AR requires a stop token to... know when to stop

		self.text_emb = Embedding(n_tokens, d_model)
		self.langs_emb = None
		self.tones_emb = None
		self.tasks_emb = None

		if self.version == 1: # legacy
			n_prom_tokens += (self.n_tasks - 1) # old models have the task tokens in the prom
			self.proms_emb = MultiEmbedding(self.n_prom_levels, n_prom_tokens, d_model)
			self.resps_emb = MultiEmbedding(self.n_resp_levels, n_resp_tokens, d_model, monolithic=self.monolithic)
		else:
			# [1024] * 8
			self.proms_emb = AudioEmbedding(
				[n_prom_tokens] * self.n_prom_levels, d_model,
				levels=self.n_prom_levels if self.version > 3 else None,
				sums=self.config.audio_embedding_sums
			)
			# [1025] + [1024] * 8
			self.resps_emb = AudioEmbedding(
				[n_resp_tokens] + [n_resp_tokens - 1] * (self.n_resp_levels - 1), d_model,
				levels=self.n_resp_levels if self.version > 3 else None,
				sums=self.config.audio_embedding_sums
			)

		
		if self.version >= 3:
			self.langs_emb = Embedding(self.n_langs, d_model) if self.n_langs > 0 else None
			self.tasks_emb = Embedding(self.n_tasks, d_model) if self.n_tasks > 0 else None
		
		if self.version >= 4:
			self.tones_emb = Embedding(self.n_tones, d_model) if self.n_tones > 0 else None

		self.sep = nn.Parameter(torch.randn(d_model))

		if self.arch_type == "transformer":
			self.sin_emb = SinusoidalEmbedding(d_model)
			self.blocks = nn.ModuleList([TransformerBlock(
				d_model=d_model,
				n_heads=n_heads,
				p_dropout=p_dropout if training else 0.0,
				causal=self.causal,
				norm_type=self.norm_type,
				n_levels=self.n_resp_levels,
			) for _ in range(n_layers) ])
		elif self.arch_type == "mistral" or self.arch_type == "mixtral":
			if n_experts <= 1:
				self.model = MistralModel(MistralConfig(
					vocab_size=n_resp_tokens,
					hidden_size=d_model,
					max_position_embeddings=75 * 60, # max-length of 60 seconds
					intermediate_size=d_model*4,
					num_hidden_layers=n_layers,
					num_attention_heads=n_heads,
					attention_dropout=p_dropout if training else 0.0,
					num_key_value_heads=n_heads,
					hidden_act="gelu",
					is_encoder_decoder=False,
					is_decoder=True,
					attn_implementation=self.config.attention if self.config is not None else None, # "flash_attention_2",
				))
			else:
				self.model = MixtralModel(MixtralConfig(
					vocab_size =n_resp_tokens,
					hidden_size=d_model,
					max_position_embeddings=75 * 60, # max-length of 60 seconds
					intermediate_size=d_model*4,
					num_hidden_layers=n_layers,
					num_attention_heads=n_heads,
					attention_dropout=p_dropout if training else 0.0,
					num_key_value_heads=n_heads,
					sliding_window=75 * 12, # 12 second context window
					output_router_logits=training,
					hidden_act="gelu",
					is_encoder_decoder=False,
					is_decoder=True,
					num_local_experts=n_experts,
					num_experts_per_tok=min(2, n_experts),
					attn_implementation=self.config.attention if self.config is not None else None, # "flash_attention_2",
				))
		elif self.arch_type == "llama":
			if n_experts <= 1:
				self.model = LlamaModel(LlamaConfig(
					vocab_size=n_resp_tokens,
					hidden_size=d_model,
					max_position_embeddings=75 * 60, # max-length of 60 seconds
					intermediate_size=d_model*4,
					num_hidden_layers=n_layers,
					num_attention_heads=n_heads,
					attention_dropout=p_dropout if training else 0.0,
					num_key_value_heads=n_heads,
					sliding_window=75 * 12, # 12 second context window
					hidden_act="gelu",
					is_encoder_decoder=False,
					is_decoder=True,
					attn_implementation=self.config.attention if self.config is not None else None, # "flash_attention_2",
				))
			else:
				self.model = MixtralModel(MixtralConfig(
					vocab_size =n_resp_tokens,
					hidden_size=d_model,
					max_position_embeddings=75 * 60, # max-length of 60 seconds
					intermediate_size=d_model*4,
					num_hidden_layers=n_layers,
					num_attention_heads=n_heads,
					attention_dropout=p_dropout if training else 0.0,
					num_key_value_heads=n_heads,
					sliding_window=75 * 12, # 12 second context window
					output_router_logits=training,
					hidden_act="gelu",
					is_encoder_decoder=False,
					is_decoder=True,
					num_local_experts=n_experts,
					num_experts_per_tok=min(2, n_experts),
					attn_implementation=self.config.attention if self.config is not None else None, # "flash_attention_2",
				))

		elif self.arch_type == "retnet":
			kwargs = dict(
				vocab_size=n_resp_tokens,
				decoder_embed_dim=d_model,
				decoder_value_embed_dim =d_model * 2,
				decoder_retention_heads=n_heads,
				decoder_ffn_embed_dim=d_model * 4,
				decoder_layers=n_layers,
				dropout=p_dropout if training else 0.0,
				checkpoint_activations=self.activation_checkpointing,
				activation_fn="gelu",
				use_layernorm=True, # self.version < 3,
				use_biases=True, # self.version < 3,
				use_glu=False, # self.version >= 3,

				chunkwise_recurrent=self.causal and self.recurrent_chunk_size > 0,
				recurrent_chunkwise_size=self.recurrent_chunk_size if self.causal else 0,
				no_output_layer=True,
				decoder_normalize_before=True,

				rotary_embedding_base=self.rotary_embedding_base, # 10000
			)

			if n_experts > 1:
				kwargs.update(dict(
					use_xmoe=True,
					moe_freq=1,
					moe_expert_count=n_experts,
					moe_gating_use_fp32=False,
				))

			self.model = RetNetDecoder(RetNetConfig(**kwargs))
		elif self.arch_type == "retnet-hf":
			kwargs = dict(
				vocab_size=n_resp_tokens,
				decoder_embed_dim=d_model,
				decoder_value_embed_dim =d_model * 2,
				decoder_retention_heads=n_heads,
				decoder_ffn_embed_dim=d_model * 4,
				decoder_layers=n_layers,
				dropout=p_dropout if training else 0.0,
				checkpoint_activations=self.activation_checkpointing,
				activation_fn="gelu",
				use_glu=False, # self.version >= 3,

				recurrent_chunk_size=self.recurrent_chunk_size if self.causal else 0,
				decoder_normalize_before=True,

				deepnorm=False,
				subln=True,
			)

			self.model = RetNetDecoder_HF(RetNetConfig_HF(**kwargs))
		elif self.arch_type == "bitnet":
			self.model = BitNetTransformer(
				num_tokens=n_resp_tokens,
				dim=d_model,
				depth=n_layers,
				heads=n_heads,
				ff_mult=4,
			)
		else:
			raise RuntimeError(f'Unknown arch specified: {self.arch_type}')

		self.classifier = nn.Linear(d_model, n_resp_tokens)

		self.accuracy_metric = MulticlassAccuracy(
			n_resp_tokens,
			top_k=10,
			average="micro",
			multidim_average="global",
			ignore_index=self.ignore_index,
		)

		self.precision_metric = MulticlassPrecision(
			n_resp_tokens,
			top_k=10,
			average="micro",
			multidim_average="global",
			ignore_index=self.ignore_index,
		)

	def _forward(
		self,
		inputs,
		mask = None,
		state = None,
	):
		x = inputs
		m = mask.squeeze(-1).int()
		aux_loss = None

		"""
		# Broken
		if state is not None and (self.arch_type == "retnet" or self.arch_type == "retnet-hf"):
			# prefill
			if len(state) == 0:
				prefill_size = x.shape[1]
				# run the initial prompt to fill the KV cache
				if self.arch_type == "retnet":
					for n in range(prefill_size):
						xi = x[:, n, :].unsqueeze(1)
						self.model(xi, incremental_state=state, token_embeddings=xi, features_only=True)
				elif self.arch_type == "retnet-hf":
					state = None
					for n in range(prefill_size):
						xi = x[:, n, :].unsqueeze(1)

						kwargs = dict(
							attention_mask=m,
							inputs_embeds=xi,
							past_key_values=state,
							use_cache=True,
							forward_impl='recurrent',
						#	return_dict=True,
						)

						out = self.model(**kwargs)
						state = out.past_key_values

			# grab last token(s)
			x = x[:, -1, :].unsqueeze(1)
		"""
		
		# HF transformer derived model
		if self.arch_type in ["llama", "mistral", "mixtral"]:
			kwargs = dict(
				attention_mask=m,
				inputs_embeds=x,
				past_key_values=state,
				use_cache=True,
			#	return_dict=True,
			)
			if self.n_experts > 1 and targ_list is not None:
				kwargs["output_router_logits"] = True

			t = self.model(**kwargs)

			x = t[0]
			
			if state is not None:
				state = t[1]
			
			if self.n_experts > 1 and targ_list is not None:
				router_logits = t[-1]
				aux_loss = self.model.config.router_aux_loss_coef * load_balancing_loss_func( router_logits, self.model.config.num_local_experts, self.model.config.num_experts_per_tok )
		elif self.arch_type == "transformer":
			# ensures we specify a quant_level for the transformer implementation's AdaLN
			l = torch.zeros((batch_size,), dtype=torch.int32) if quant_levels is None else quant_levels
			l = l.to(device)
			# inject position information
			x = self.sin_emb.add_pe(x)
			# pass our inputs through the transformer
			for block in self.blocks:
				x = block(x, m, l)
		elif self.arch_type == "retnet":
			# pass our inputs through the RetNet
			x, _ = self.model(x, incremental_state=state, token_embeddings=x, features_only=True)
			if _ is not None and "l_aux" in _ and self.n_experts > 1:
				aux_loss = torch.sum(torch.stack([ t for t in _["l_aux"] if t is not None])) * 0.001
		elif self.arch_type == "retnet-hf":
			first = state is None or len(state) == 0

			kwargs = dict(
				attention_mask=m,
				inputs_embeds=x if first else x[:, -1, :].unsqueeze(1),
				past_key_values=None if first else state,
				use_cache=True,
				forward_impl='parallel' if first else 'recurrent',
				return_dict=True,
			)

			out = self.model(**kwargs)
			x = out.last_hidden_state
			if state is not None:
				state = out.past_key_values
				
		elif self.arch_type == "bitnet":
			x = self.model(x)

		# output projection layer with masking
		x = self.classifier(x) * mask

		return x, state, aux_loss

	def inputs(
		self,
		text_list: list[Tensor],
		proms_list: list[Tensor],
		resps_list: list[Tensor],
		targ_list: list[Tensor] | None = None,

		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
	):
		device = text_list[0].device
		batch_size = len(text_list)

		inputs = [ [] for _ in range(batch_size) ]
		for i in range(batch_size):
			if text_list is not None:
				inputs[i].append( ( "text", text_list[i] ) )
			if proms_list is not None:
				inputs[i].append( ( "prom", proms_list[i] ) )
			if resps_list is not None:
				inputs[i].append( ( "resp", resps_list[i] ) )
			if targ_list is not None:
				inputs[i].append( ( "targ", targ_list[i] ) )

		return inputs

	def inputs_to_embeddings(
		self,
		inputs: list,
		quant_levels: Tensor | None = None
	):
		x_list = []
		for b_i in range(len(inputs)):
			batch = []
			for i in range(len(inputs[b_i])):
				name, input = inputs[b_i][i]
				embedding = None
				if name == "text":
					embedding = self.text_emb( input )
				elif name == "lang":
					embedding = self.langs_emb( input )
				elif name == "prom":
					embedding = self.proms_emb( input )
				elif name == "tone":
					embedding = self.tones_emb( input )
				elif name == "resp":
					embedding = self.resps_emb( input, quant_levels[b_i] if quant_levels is not None else None )
				else:
					continue

				batch.append(embedding)
	
			x_list.append( _join( batch, self.sep ) )

		return x_list

	def training_targets(
		self,
		inputs: list,
	):
		x_list = []
		for bi in range(len(inputs)):
			batch = []
			for i in range(len(inputs[bi])):
				name, input = inputs[bi][i]
				device = input.device

				if name == "prom":
					batch.append( torch.full_like(input[..., 0], self.ignore_index) )
				elif name in ["text", "lang", "tone", "targ"]:
					batch.append( input )

			x_list.append( _join( batch, torch.tensor(self.ignore_index, device=device) ) )

		return x_list

	def forward(
		self,
		inputs: list,

		quant_levels: Tensor | None = None,
		state: dict | list | None = None,
	):

		x_list = self.inputs_to_embeddings( inputs, quant_levels )
		x, m = list_to_tensor(x_list)

		# yes, there's a better way.
		training = False
		for b_i in range(len(inputs)):
			for i in range(len(inputs[b_i])):
				name, input = inputs[b_i][i]
				if name == "targ":
					training = True


		device = x.device
		batch_size = len(x_list)
		
		# pad our input and mask, but retain the original length by doing it after
		if self.l_padding and x.shape[1] % self.l_padding != 0:
			# pad input
			shape = list(x.shape)
			shape[1] = self.l_padding - shape[1] % self.l_padding

			padding = torch.zeros(shape, dtype=x.dtype, device=x.device)
			x = torch.cat([x, padding], dim=1)

			# pad mask
			shape[2] = 1
			padding = torch.zeros(shape, dtype=x.dtype, device=x.device)
			m = torch.cat([m, padding], dim=1)


		x, state, aux_loss = self._forward(
			inputs=x,
			mask=m,
			state=state,
		)

		# Remove padding
		logits = [ hi[:li] for hi, li in zip(x, map(len, x_list)) ]
		
		# compute loss if the target is given
		if training:
			target_list = self.training_targets( inputs )
			
			# modify only for the AR so it can properly behave like a transformer
			for i in range(len(target_list)):
				if quant_levels is not None and quant_levels[i] > 0:
					continue

				logits[i] = logits[i][..., :-1, :] # shift the target so that token n...
				target_list[i] = target_list[i][..., 1:] # predicts token n + 1

			target = torch.cat( target_list )
			inputs = torch.cat( logits )

			self.loss = dict(
				# "nll" was in the original implementation and should actually just be called something else
				nll = F.cross_entropy( inputs, target, ignore_index=self.ignore_index )
			)
			self.stats = dict(
				acc = self.accuracy_metric( inputs, target ),
				# precision = self.precision_metric( inputs, target ),
			)

			if aux_loss is not None:
				self.loss["nll"] += aux_loss
			
		return (logits, state) if state is not None else logits

	def sample(
		self,
		logits: list[Tensor],
		resps_list: list[Tensor],
		quant_levels: Tensor | None = None,

		temperature: float = 1.0,
		min_temperature: float = -1.0,
		top_k: int = -100,
		top_p: float = 1.0,

		repetition_penalty: float = 1.0,
		repetition_penalty_decay: float = 0.0,
		
		length_penalty: float = 0.0,
		
		beam_width: int = 0,

		mirostat: list[dict] | None = None,
	):
		if min_temperature < 0:
			min_temperature = temperature
		# (NAR) return the entire generated response
		if quant_levels is not None:
			logits = [ logit[-l:] for logit, l in zip(logits, map(len, resps_list)) ]
		# (AR chunkwise) return the last chunkwise piece
		elif self.causal and self.recurrent_chunk_size > 0:
			logits = [ logit[-l:] for logit, l in zip(logits, self.recurrent_chunk_size) ]
		# (AR) return just the last code
		else:
			logits = [ logit[-1:] for logit in logits ]

		devices = [ logit.device for logit in logits ]
		logits = [ logit.to(device="cpu", dtype=logit.dtype if logit.dtype != torch.float16 else torch.float32) for logit in logits ]

		# perform repetition penalizing	
		logits = [ reptition_penalize(logit, previous=resps[:, -1], factor=repetition_penalty, decay=repetition_penalty_decay) for logit, resps in zip( logits, resps_list ) ]

		# (AR) perform length penalizing
		if quant_levels is None and self.causal:
			logits = [ length_penalize(logit, length=l + 1, factor=length_penalty, token=self.stop_token) for logit, l in zip( logits, map(len, resps_list) ) ]

		# perform top_k/top_p filtering of our logits
		if top_k > 0 or top_p < 1.0:
			logits = [ top_k_top_p_filtering(logit, top_k=top_k, top_p=top_p) for logit in logits ]	

		# trigger dynamic temperature sampling if the minimum temperature is not the same as the sampling temperature
		#	 epsilon float comparison because I don't trust Python
		if abs(temperature - min_temperature) >= 0.001: 
			logits = [ dynamic_temperature(logit, temperature=temperature, min_temperature=min_temperature) for logit in logits ]
		else:
			logits = [ logit / temperature for logit in logits ]

		# do mirostat sampling
		# currently incompatible with beam searching with the way the two are implemented, perhaps a night of brain bashing can make the two work
		if mirostat is not None:
			# mirostat sampling
			return [ mirostat_sample(logit, state=state) for logit, state in zip(logits, mirostat) ]

		# do beam search (naive implementation)
		# picks the top-k across all batches, and re-batches those resultant tokens
		# returns the logit scores as well to be P-concatted with the previous scores
		# to-do: not naively implement beam searching
		if beam_width > 1:
			candidates = top_k_logits_list( logits, beam_width )
			res = [ torch.tensor(token, dtype=torch.int16).unsqueeze(dim=-1) for batch, token in candidates ]
			scores = [ logits[batch].flatten()[token] for batch, token in candidates ]
			return res, scores

		# and sample
		return [ Categorical(logits=logit).sample() for logit in logits ]

def example_usage():
	from ..config import cfg
	cfg.trainer.backend = "local"
	cfg.trainer.check_for_oom = False

	from functools import partial

	from einops import repeat

	from ..emb.qnt import decode_to_file
	from ..engines import Engine, Engines
	from tqdm import tqdm, trange
	from ..utils import wrapper as ml

	from .ar import AR
	from .nar import NAR

	device = "cuda"
	x8 = partial(repeat, pattern="t -> t l", l=cfg.model.prom_levels) 
	symmap = {'<s>': 1, '</s>': 2, ' ': 3, '.': 4, ',': 5, '!': 6, '?': 7, 'p': 7, 'iː': 8, 'ɚ': 9, 'ˌ': 10, 'dˌ': 11, 'mˌ': 12, 'd': 13, 'ɹ': 14, 'tˈ': 15, 'pˌ': 16, 'uː': 17, 'l': 18, 'æ': 19, 'ɛ': 20, 'ɪ': 21, 'j': 22, 'ʊ': 23, 't': 24, 'n': 25, 'v': 26, 'a': 27, 'o': 28, 'ŋ': 29, 'w': 30, 'ʌ': 31, 'hˈ': 32, 'ɡˈ': 33, 'ə': 34, 'θˈ': 35, 'dˈ': 36, 'wˌ': 37, 'h': 38, 'z': 39, 'k': 40, 'ð': 41, 'ɡˌ': 42, 'ˈ': 43, 'fˈ': 44, 'i': 45, 's': 46, 'ʃ': 47, 'wˈ': 48, 'ðˈ': 49, 'ɹˈ': 50, 'lˈ': 51, 'ɡ': 52, 'oː': 53, 'mˈ': 54, 'e': 55, 'ɑː': 56, 'nˈ': 57, 'm': 58, 'θˌ': 59, 'sˈ': 60, 'f': 61, 'ɔː': 62, 'hˌ': 63, 'b': 64, 'jˈ': 65, 'ɐ': 66, 'ʒˈ': 67, 'θ': 68, 'bˈ': 69, 'ɾ': 70, 'ɜː': 71, 'ʌˈ': 72, 'ʃˌ': 73, 'bˌ': 74, 'kˈ': 75, 'ɔ': 76, 'zˈ': 77, 'ᵻ': 78, 'kˌ': 79, 'vˈ': 80, 'fˌ': 81, 'ʒ': 82, 'ʃˈ': 83, 'ɹˌ': 84, 'tˌ': 85, 'pˈ': 86, 'ðˌ': 87, 'sˌ': 88, 'nˌ': 89, 'lˌ': 90, '̩': 91, 'ʔ': 92, 'vˌ': 93, 'ɪˈ': 94, '"': 95, 'ɪˌ': 96, 'ʒˌ': 97, 'uːˌ': 98, 'ʊˈ': 99, 'jˌ': 100, 'uːˈ': 101, 'iːˈ': 102, 'zˌ': 103, '.ˈ': 104, '…': 105, 'ŋˌ': 106, 'ɐˌ': 107, '—ˈ': 108, 'iˌ': 109, 'iːˌ': 110, 'ɛː': 111, ')': 112, ')ˈ': 113, '(': 114, 'u': 115, '-': 116, 'ɖˈ': 117, 'iˈ': 118, 'ʰˈ': 119, 'ɟˈ': 120, '̃': 121, 'eː': 122, 'ɾˈ': 123, 'r': 124, 'ʰ': 125, '-ˌ': 126, 'ɫ': 127, 'q': 128, '—': 129, 'ʊˌ': 130, 'aː': 131, 'cˈ': 132, '…ˈ': 133, 'c': 134, 'ɳ': 135, 'ɐˈ': 136, 'x': 137, 'ʔˌ': 138, '.ˌ': 139, 'ɑ': 140, '?ˈ': 141, '̩ˈ': 142, '"ˈ': 143, ',ˈ': 144, 'ŋˈ': 145, 'əˌ': 146, '!ˈ': 147, '"ˌ': 148, '?ˌ': 149, ',ˌ': 150, '—ˌ': 151, '̩ˌ': 152, 'əˈ': 153, '!ˌ': 154, 'ɬ': 155, 'ʲ': 156, '¡': 157, 'ɯ': 158, 'qˌ': 159, 'ʑ': 160, 'ʑˈ': 161, '¿': 162, 'ɑːˈ': 163, 'iːː': 164, 'ɛˈ': 165, '¡ˈ': 166, 'æˈ': 167, 'ç': 168, 'ɾˌ': 169, 'ᵻˈ': 170, 'xˈ': 171, 'ɔːˈ': 172, ';': 173, 'ɬˌ': 174, ':': 175, 'ʔˈ': 176, 'ɑːˌ': 177, 'ɬˈ': 178}
	def tokenize(content, lang_marker="en"):
		split = content.split(" ")
		phones = [f"<s>"] + [ " " if not p else p for p in split ] + [f"</s>"]
		return torch.tensor([*map(symmap.get, phones)]).to()

	kwargs = {
		'n_tokens': 1024,
		'd_model': 1024,
		'n_heads': 16,
		'n_layers': 12,
	}
	models = { "ar": AR(**kwargs).to(device), "nar": NAR(**kwargs).to(device) }
	
	for name, model in models.items():
		print(f"{name} parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

	engines = Engines({ name: Engine(model=model, optimizer=ml.AdamW(model.parameters(), lr=1e-4)) for name, model in models.items() })

	train = True

	qnt = torch.load("data/qnt.pt")[0].t()[:, :cfg.model.prom_levels].to(device)
	text_list = [
		tokenize("ˈ a ɪ   w ɪ l   nˌ ɑː t  ˈ æ s k   ɐ   sˈ ɛ k ə n d   tˈ a ɪ m").to(device),
		#tokenize("ˌ ɔ n   ɡˌ o ʊ ɪ ŋ   hˈ o ʊ m   ð ə   tˈ uː   f ɹˈ ɛ n d z   fˈ a ʊ n d   ɐ   lˈ ɛ ɾ ɚ   f ɹ ʌ m  ˈ æ θ o ʊ z ,   hˌ uː   d ɪ zˈ a ɪ ɚ d   ðˌ ɛ m   t ə   mˈ iː t   hˌ ɪ m   æ t   ð ə   ɡ ɹˈ æ n d   t ʃˈ ɑː ɹ l ɪ mˌ æ ɡ n i   ɔ n ð ə   fˈ ɑː l o ʊ ɪ ŋ   dˈ e ɪ .").to(device),
	]

	proms_list = [
		qnt.to(device),
	]
	resps_list = [
		qnt.to(device),
	]
	
	def sample( name, steps=600 ):
		AR = None
		NAR = None

		engines.eval()
		for name, engine in engines.items():
			if name[:2] == "ar":
				AR = engine
			elif name[:3] == "nar":
				NAR = engine

		resps_list = AR(text_list, proms_list, max_steps=steps, sampling_temperature=1.0)
		resps_list = [r.unsqueeze(-1) for r in resps_list]		
		codes = NAR( text_list, proms_list, resps_list=resps_list, sampling_temperature=0.2 ) 

		decode_to_file(resps_list[0], f"./data/ar.{name}.wav", device=device)
		decode_to_file(codes[0], f"./data/ar+nar.{name}.wav", device=device)
	
	if train:
		sample("init", 15)

		engines.train()
		t = trange(500)
		for i in t:
			stats = {"step": i}
			"""
			for name, engine in engines.items():
				stats |= engine.traverse(text_list=text_list, proms_list=proms_list, resps_list=resps_list)
			"""
			stats = engines.step({"text_list": text_list, "proms_list": proms_list, "resps_list": resps_list})
			tqdm.write(f"{stats}")
	else:
		for name, engine in engines.items():
			engine.module.load_state_dict(torch.load(f"./data/{name}.pth"))

	sample("final")
	

if __name__ == "__main__":
	example_usage()
