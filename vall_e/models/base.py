import math
import torch
import torch.nn.functional as F
import traceback
import numpy as np

from typing import Literal, overload
from functools import partial
from einops import rearrange

from torch import Tensor, einsum, nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy, MulticlassPrecision

from .retnet import RetNetDecoder, RetNetConfig
from .transformer import SinusoidalEmbedding, Block as TransformerBlock

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

# Simple filter to modify a token's probability if it shows up in the past
# `one_time` will only apply the penalty once
# `decay` is a factor that will exponentially apply to how far away it is
def reptition_penalize( logits, previous, factor=1.0, decay=0.0, one_time=True ):
	if factor == 1.0 or previous is None:
		return logits

	unique = set()
	priors = reversed(previous.tolist())
	for distance, token in enumerate(priors):
		# skip if we're only applying the decay once
		if one_time and token in unique:
			continue

		distance += 1
		logits[:, token] /= factor * (distance ** decay)
		
		# add to set if we care about it
		if one_time:
			unique.add(token)

	return logits

# Simple "filter" that modifies the logit for the stop token, based on the sequence length
# `length` is the length of the sequence currently
# `factor` is the power the length is raised to, so values > 0 will yield longer sequences, values < 0 will yield shorter sequences
# `token` is the stop token.
def length_penalize( logits, length, factor=0.0, token=-1 ):
	if factor == 0.0:
		return logits

	logits[:, token] /= (length ** factor)
	return logits

# Credit to https://github.com/microsoft/unilm/blob/master/xtune/src/transformers/modeling_utils.py#L1145 / https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
def top_k_top_p_filtering( logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens=1 ):
	"""Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
	Args:
		logits: logits distribution shape (batch size, vocabulary size)
		if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
		if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
			Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
		Make sure we keep at least min_tokens per batch example in the output
	"""
	if top_k > 0:
		top_k = min(max(top_k, min_tokens), logits.size(-1))  # Safety check
		# Remove all tokens with a probability less than the last token of the top-k
		indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
		logits[indices_to_remove] = filter_value

	if top_p < 1.0:
		sorted_logits, sorted_indices = torch.sort(logits, descending=True)
		cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

		# Remove tokens with cumulative probability above the threshold (token with 0 are kept)
		sorted_indices_to_remove = cumulative_probs > top_p
		if min_tokens > 1:
			# Keep at least min_tokens (set to min_tokens-1 because we add the first one below)
			sorted_indices_to_remove[..., :min_tokens] = 0
		# Shift the indices to the right to keep also the first token above the threshold
		sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
		sorted_indices_to_remove[..., 0] = 0

		# scatter sorted tensors to original indexing
		indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
		logits[indices_to_remove] = filter_value

	return logits

# picks the top K tokens amongst a batch of logits
# logits: [Tensor] list of logits
# candidates: [(batch, token)] list, where batch indicates the index of the logits the given token is from
def top_k_logits_list( logits_list, k ):
	# ( batch, tokens ) => ( batch x tokens )
	logits = torch.cat( logits_list )
	candidates = list(torch.topk(logits.flatten(), k).indices.tolist()) # perform top-k across all logits
	for i, index in enumerate(candidates):
		t = []
		N = np.prod(logits.size())
		for n in logits.size():
			N //= n
			t.append(index // N)
			index %= N
		candidates[i] = tuple(t)
	return candidates


# Credit to: https://github.com/basusourya/mirostat/
# performs mirostat-based sampling
# logits: Tensor of logit probabilities
# state: the mirostat state
def mirostat_sample( logits, state = None ):
	def compute_k(prob, n, tau):
		num = 0
		den = 0
		for i in range(100):
			b = prob[i]/prob[i+1]
			t = (i+2)/(i+1)
			num += math.log(b)*math.log(t)
			den += math.log(t)**2
				
		s = num/den
		eps = s-1
		k = ((eps*(2**(tau)))/(1-n**(-eps)))**(1/s)
		k = round(k)
		return k

	if "max_surprise" not in state:
		state["max_surprise"] = state["tau"] * 2

	if "error_surprise" not in state:
		state["error_surprise"] = 0

	if "running_total_surprise" not in state:
		state["running_total_surprise"] = 0
	
	sorted_logits, sorted_indices = torch.sort( logits[-1, :], descending=True )
	prob_original = torch.softmax( sorted_logits, dim=-1 ).tolist()

	k = compute_k(prob_original, state["n"], state["max_surprise"]) + 1

	sorted_logits = sorted_logits[0:k]
	sorted_indices = sorted_indices[0:k]
	prob_topk = torch.softmax(sorted_logits, dim = 0)
	prev_i = torch.multinomial(prob_topk, num_samples=1, replacement=True)
	
	state["index_surprise"] = math.log2(1/prob_original[prev_i])
	state["running_total_surprise"] += state["index_surprise"]
	state["error_surprise"] = state["index_surprise"] - state["tau"]
	state["max_surprise"] -= state["eta"] * state["error_surprise"]
	state["token"] = sorted_indices[prev_i]

	return state


# automagically parses a batch-list and returns it as a list
class Embedding(nn.Embedding):
	def forward(self, x_list: list[Tensor]) -> list[Tensor]:
		if len(x_list) == 0:
			return []

		return super().forward(torch.cat(x_list)).split([*map(len, x_list)])

class MultiEmbedding(nn.Module):
	"""
	This embedding sums embeddings on different levels.
	"""

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
	def __init__(self, l_tokens, token_dim):
		super().__init__()
		self.embeddings = nn.ModuleList([nn.Embedding(n_tokens, token_dim) for n_tokens in l_tokens])
	
	def forward(self, x_list: list[Tensor], quant_levels: Tensor | None = None ) -> list[Tensor]:
		res_list = []

		for i, xi in enumerate(x_list):
			# prom
			if quant_levels is None and xi.shape[-1] > 1:
				x = sum( [ self.embeddings[k]( xi[:, k] ) for k in range(xi.shape[-1]) ] )
			# AR resp
			elif quant_levels is None or quant_levels[i] == 0:
				x = self.embeddings[0]( xi[:, 0] )
			# NAR resp
			else:
				x = sum( [ self.embeddings[k+1]( xi[:, k] ) for k in range(xi.shape[-1]) ] )
			res_list.append(x)

		return res_list

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

	@staticmethod
	def _samplewise_merge_tensors(*l, sep: Tensor | None):
		if sep is None:
			cat = torch.cat
		else:
			cat = partial(_join, sep=sep)
		return [*map(cat, zip(*l))]

	def __init__(
		self,
		n_tokens: int = 1024,
		d_model: int = 512,
		n_heads: int = 8,
		n_layers: int = 12,
		p_dropout: float = 0.1,

		config = None, 
	):
		super().__init__()
		self.config = config
		self.activation_checkpointing = self.config.activation_checkpointing if self.config is not None else True

		self.n_tokens = n_tokens
		self.d_model = d_model
		self.n_heads = n_heads
		self.n_layers = n_layers

		# +1 to include the stop token
		# to-do: undo this dogshit mistake; tasks tokens should be delegated to its own embedding
		n_prom_tokens = n_tokens
		n_resp_tokens = n_tokens + (1 if self.causal else 0) # AR requires a stop token to... know when to stop

		self.text_emb = Embedding(n_tokens, d_model)

		if self.version == 1: # legacy
			n_prom_tokens += (self.n_tasks - 1) # old models have the task tokens in the prom
			self.proms_emb = MultiEmbedding(self.n_prom_levels, n_prom_tokens, d_model)
			self.resps_emb = MultiEmbedding(self.n_resp_levels, n_resp_tokens, d_model, monolithic=self.monolithic)
		else:
			# [1024] * 8
			self.proms_emb = AudioEmbedding([n_prom_tokens] * self.n_prom_levels, d_model)
			# [1025] + [1024] * 8
			self.resps_emb = AudioEmbedding([n_resp_tokens] + [n_resp_tokens - 1] * (self.n_resp_levels - 1), d_model)

			# self.langs_emb = Embedding(self.n_langs, d_model)
			# self.tasks_emb = Embedding(self.n_tasks, d_model)

		self.sep = nn.Parameter(torch.randn(d_model))

		if self.arch_type == "transformer":
			self.sin_emb = SinusoidalEmbedding(d_model)
			self.blocks = nn.ModuleList([TransformerBlock(
				d_model=d_model,
				n_heads=n_heads,
				p_dropout=p_dropout,
				causal=self.causal,
				norm_type=self.norm_type,
				n_levels=self.n_resp_levels,
			) for _ in range(n_layers) ])
		elif self.arch_type == "retnet":
			self.retnet = RetNetDecoder(RetNetConfig(
				vocab_size=n_tokens,
				decoder_embed_dim=d_model,
				decoder_value_embed_dim =d_model * 2,
				decoder_retention_heads=n_heads,
				decoder_ffn_embed_dim=d_model * 4,
				decoder_layers=n_layers,
				dropout=p_dropout,
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
			))

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

	def forward(
		self,
		text_list: list[Tensor],
		proms_list: list[Tensor],
		resps_list: list[Tensor],
		targ_list: list[Tensor] | None = None,

		#langs_list: list[Tensor] | None = None,
		#tasks_list: list[Tensor] | None = None,

		quant_levels: Tensor | None = None,
		state: dict | None = None,
	):
		x_list = self._samplewise_merge_tensors(
			self.text_emb(text_list),
			#self.langs_emb(langs_list),
			self.proms_emb(proms_list),
			#self.tasks_emb(tasks_list),
			self.resps_emb(resps_list, quant_levels),
			sep=self.sep,
		)

		x, m = list_to_tensor(x_list)
		
		batch_size = len(text_list)
		device = x.device

		if state is not None and self.arch_type == "retnet":
			# prefill
			if len(state) == 0:
				prefill_size = x.shape[1]
				# run the initial prompt to fill the KV cache
				for n in range(prefill_size):
					xi = x[:, n, :].unsqueeze(1)
					self.retnet(xi, incremental_state=state, token_embeddings=xi, features_only=True)

			# grab last token(s)
			x = x[:, -1, :].unsqueeze(1)

		if self.arch_type == "transformer":
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
			x, _ = self.retnet(x, incremental_state=state, token_embeddings=x, features_only=True)
		
		# output projection layer with masking
		x = self.classifier(x) * m

		# Remove padding
		logits = [ hi[:li] for hi, li in zip(x, map(len, x_list)) ]

		# compute loss if the target is given
		if targ_list is not None:
			ignore_sep = torch.tensor(self.ignore_index, device=device)
			# create a tensor sequence with one RVQ-bin of the input prompt, but with `ignore_index`, as the prompt is not neeeded for computing the loss against
			prom_list = [ torch.full_like(t[..., 0], self.ignore_index) for t in proms_list ]
			# remake input sequence
			text_prom_list = self._samplewise_merge_tensors( text_list, prom_list, sep=ignore_sep )

			# process each batch
			for i in range(len(text_prom_list)):
				# for the AR and NAR, shift the text/input prompt into the future by 1, and ignore the rolled back token
				text_prom_list[i] = text_prom_list[i].roll(-1, dims=0)
				text_prom_list[i][-1] = self.ignore_index

				# for the AR, shift the target response into the future by 1, and ignore the rolled back text token
				if quant_levels is None or quant_levels[i] == 0:
					targ_list[i] = targ_list[i].clone().roll(-1, dims=0) # clone ensures it's not an aliased copy/view of resps
					targ_list[i][-1] = self.stop_token

			# create the new target sequence to compute the loss against
			target = torch.cat( self._samplewise_merge_tensors( text_prom_list, targ_list, sep=ignore_sep ) )
			inputs = torch.cat( logits )

			self.loss = dict(
				# "nll" was in the original implementation and should actually just be called something else
				nll = F.cross_entropy( inputs, target, ignore_index=self.ignore_index )
			)
			self.stats = dict(
				acc = self.accuracy_metric( inputs, target ),
				precision = self.precision_metric( inputs, target ),
			)
			
		return logits

	def sample(
		self,
		logits: list[Tensor],
		resps_list: list[Tensor],
		quant_levels: Tensor | None = None,

		temperature: float = 1.0,
		top_k: int = -100,
		top_p: float = 1.0,

		repetition_penalty: float = 1.0,
		repetition_penalty_decay: float = 0.0,
		
		length_penalty: float = 0.0,
		
		beam_width: int = 0,

		mirostat: list[dict] | None = None,
	):
		# (NAR) return the entire generated response
		if quant_levels is not None:
			logits = [ logit[-l:] for logit, l in zip(logits, map(len, resps_list)) ]
		# (AR chunkwise) return the last chunkwise piece
		elif self.causal and self.recurrent_chunk_size > 0:
			logits = [ logit[-l:] for logit, l in zip(logits, self.recurrent_chunk_size) ]
		# (AR) return just the last code
		else:
			logits = [ logit[-1:] for logit in logits ]

		# perform repetition penalizing	
		logits = [ reptition_penalize(logit, previous=resps[:, -1], factor=repetition_penalty, decay=repetition_penalty_decay) for logit, resps in zip( logits, resps_list ) ]

		# (AR) perform length penalizing
		if quant_levels is None and self.causal:
			logits = [ length_penalize(logit, length=l + 1, factor=length_penalty, token=self.stop_token) for logit, l in zip( logits, map(len, resps_list) ) ]

		# scale our logits by the temp
		logits = [ logit / temperature for logit in logits ]

		# perform top_k/top_p filtering of our logits
		if top_k > 0 or top_p < 1.0:
			logits = [ top_k_top_p_filtering(logit, top_k=top_k, top_p=top_p) for logit in logits ]	

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
			res = [ torch.tensor(token, device=logits[batch].device, dtype=torch.int16).unsqueeze(dim=-1) for batch, token in candidates ]
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
	x8 = partial(repeat, pattern="t -> t l", l=cfg.models.prom_levels) 
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

	qnt = torch.load("data/qnt.pt")[0].t()[:, :cfg.models.prom_levels].to(device)
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
