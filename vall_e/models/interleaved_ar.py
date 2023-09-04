import math
import torch
import torch.nn.functional as F
import traceback

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

try:
	from ..ext.interleaver import (
		CodebooksPatternProvider,
		DelayedPatternProvider,
		MusicLMPattern,
		ParallelPatternProvider,
		UnrolledPatternProvider,
		VALLEPattern,
	)
except Exception as e:
	pass

from ..config import cfg

def _get_pattern_provider( name ):
	return {
		'parallel': ParallelPatternProvider,
		'delay': DelayedPatternProvider,
		'unroll': UnrolledPatternProvider,
		'valle': VALLEPattern,
		'musiclm': MusicLMPattern,
	}[name]

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

class Embedding(nn.Embedding):
	def forward(self, x_list: list[Tensor]) -> list[Tensor]:
		if len(x_list) == 0:
			return []

		return super().forward(torch.cat(x_list)).split([*map(len, x_list)])


class MultiEmbedding(nn.Embedding):
	"""
	This embedding sums embeddings on different levels.
	"""

	def __init__(self, max_n_levels, n_tokens, token_dim):
		super().__init__(max_n_levels, token_dim)
		self.max_n_levels = max_n_levels
		self.n_tokens = n_tokens
		self.weight = nn.Parameter(torch.randn(max_n_levels, n_tokens, token_dim))

	def forward(self, x_list: list[Tensor]) -> list[Tensor]:
		if len(x_list) == 0:
			return []

		w = self.weight

		padded_x_list = []

		for xi in x_list:
			xi = F.one_hot(xi.to(torch.int64), num_classes=self.n_tokens)  # t l' k
			xi = F.pad(xi, (0, 0, 0, w.shape[0] - xi.shape[1]))  # t l k
			padded_x_list.append(xi.to(w))

		x = torch.cat(padded_x_list)  # n l k
		x = einsum("l k d, n l k -> n d", w, x)

		x_list = x.split([*map(len, x_list)])

		return x_list


class Base(nn.Module):
	@property
	def causal(self):
		return True

	@property
	def use_stop_token(self):
		return True

	@property
	def norm_type(self):
		return "ln"

	@property
	def arch_type(self) -> str:
		return "retnet"

	@property
	def n_prom_levels(self) -> int:
		return 4

	@property
	def n_resp_levels(self) -> int:
		return 4

	@property
	def n_max_levels(self) -> int:
		return 4

	@property
	def n_tasks(self) -> int:
		return 16

	@property
	def resp_loss_only(self) -> bool:
		return False

	@property
	def recurrent_chunk_size(self) -> int:
		return 0

	@property
	def interleave_pattern(self) -> str | None:
		return "musiclm"

	@property
	def stop_token(self):
		return self.n_tokens + 0

	@property
	def interleaved_token(self):
		return self.n_tokens + 1

	@property
	def ignore_index(self):
		return -100 # self.interleaved_token

	def _prune(self, l: Tensor):
		indices = (l == self.stop_token).nonzero()
		if len(indices) == 0:
			return l
		return l[: indices.min().item()]

	@staticmethod
	def _unsqueeze_list(x_list, axis=-1):
		return [x.unsqueeze(dim=axis) for x in x_list]

	@staticmethod
	def _samplewise_merge_tensors(*l, sep: Tensor | None):
		if sep is None:
			cat = torch.cat
		else:
			cat = partial(_join, sep=sep)
		return [*map(cat, zip(*l))]

	def _interleave( self, codes ):
		if not self.interleave_pattern:
			return codes

		return codes.flatten()
		"""
		pattern_provider = _get_pattern_provider( self.interleave_pattern )( self.n_resp_levels )
		pattern = pattern_provider.get_pattern( codes.shape[0] )
		res, _, _ = pattern.build_pattern_sequence( codes.t()[None, :, :], self.interleaved_token, keep_only_valid_steps=True )
		return res[0].t().flatten()
		"""

	def _deinterleave( self, codes ):
		if not self.interleave_pattern:
			return codes

		return torch.unflatten( codes[:codes.shape[0] // self.n_resp_levels * self.n_resp_levels], 0, ( codes.shape[0] // self.n_resp_levels, self.n_resp_levels ) )
		"""
		if codes.dim() == 1:
			codes = torch.unflatten( codes[:codes.shape[0] // self.n_resp_levels * self.n_resp_levels], 0, ( codes.shape[0] // self.n_resp_levels, self.n_resp_levels ) )

		pattern_provider = _get_pattern_provider( self.interleave_pattern )( self.n_resp_levels )
		pattern = pattern_provider.get_pattern( codes.shape[0] )
		res, _, _ = pattern.revert_pattern_sequence( codes, special_token=self.interleaved_token)
		return res[0].t()
		"""

	def __init__(
		self,
		n_tokens: int = 1024,
		d_model: int = 512,
		n_heads: int = 8,
		n_layers: int = 12,
		p_dropout: float = 0.1,

		config: dict | None = None
	):
		super().__init__()
		self._cfg = config
		self.n_tokens = n_tokens
		self.d_model = d_model
		self.n_heads = n_heads
		self.n_layers = n_layers

		# + tasks for each token they represent in the prom
		n_prom_tokens = n_tokens + (self.n_tasks - 1) + (1 if self.interleave_pattern else 0) # - 1 because tts is an inherent task
		# +1 to include the stop token + 1 to include interleave token
		n_resp_tokens = n_tokens + (1 if self.use_stop_token else 0) + (1 if self.interleave_pattern else 0) # AR requires a stop token to... know when to stop

		self.text_emb = Embedding(n_tokens, d_model)
		self.proms_emb = MultiEmbedding(self.n_prom_levels, n_prom_tokens, d_model)
		self.resps_emb = MultiEmbedding(1, n_resp_tokens, d_model)

		self.sep = nn.Parameter(torch.randn(d_model))

		if self.arch_type == "transformer":
			self.sin_emb = SinusoidalEmbedding(d_model)
			self.blocks = nn.ModuleList([TransformerBlock(
				d_model=d_model,
				n_heads=n_heads,
				p_dropout=p_dropout,
				causal=self.causal,
				norm_type=self.norm_type,
				n_levels=1,
			) for _ in range(n_layers) ])

		elif self.arch_type == "retnet":
			self.retnet = RetNetDecoder(RetNetConfig(
				vocab_size=n_tokens,
				decoder_embed_dim=d_model,
				decoder_retention_heads=n_heads,
				decoder_ffn_embed_dim=d_model * 4,
				decoder_layers=n_layers,
				dropout=p_dropout,
				checkpoint_activations=True,

				chunkwise_recurrent=self.causal and self.recurrent_chunk_size > 0,
				recurrent_chunkwise_size=self.recurrent_chunk_size if self.causal else 0,
				no_output_layer=True,
				decoder_normalize_before=True,
			))

		# I imagine because each step returns `resp_level`s tokens at once, so we need to have a classifier for each level
		#self.classifier = nn.ModuleList([ nn.Linear(d_model, n_resp_tokens) for _ in range(self.n_resp_levels) ]) if self.interleave_pattern else nn.Linear(d_model, n_resp_tokens)
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

	@overload
	def forward(
		self,
		text_list: list[Tensor],
		proms_list: list[Tensor],
		resps_list: list[Tensor],
		targ_list: list[Tensor] | None = None,
		quant_levels: Tensor | None = None,
		shift_targ_list: bool = False,
		return_all: Literal[False] = False,
		return_all_resp: Literal[False] = False,
		sampling_temperature: float = 1.0,
	) -> Tensor:
		...

	@overload
	def forward(
		self,
		text_list: list[Tensor],
		proms_list: list[Tensor],
		resps_list: list[Tensor],
		targ_list: list[Tensor] | None = None,
		quant_levels: Tensor | None = None,
		shift_targ_list: bool = False,
		return_all: Literal[True] = True,
		return_all_resp: Literal[True] = True,
		sampling_temperature: float = 1.0,
	) -> list[Tensor]:
		...

	def _forward(
		self,
		text_list: list[Tensor],
		proms_list: list[Tensor],
		resps_list: list[Tensor],
		targ_list: list[Tensor] | None = None,
		quant_levels: Tensor | None = None,
		shift_targ_list: bool = False,
		return_all: bool = False,
		return_all_resp: bool = False,
		sampling_temperature: float = 1.0,

		state: dict | None = None,
	):
		"""
		Args:
			text_list: [t] * b
			proms_list: [t' l] * b, l quantization levels.
			resps_list: [t'' l] * b, l quantization levels.
			targ_list: [t''] * b, one quantization level only; when given, loss will be computed
			quant_levels: specify which quant_levels to feed forward, used in NAR mode.
			shift_targ_list: whether to shift target list when computing loss. True if AR.
			return_all_resp: True if NAR.
			sampling_temperature: a lower temperature makes the result more robust but less diverse.
		Returns:
			y: sampled tokens
		"""

		batch_size = len(text_list)

		x_list = self._samplewise_merge_tensors(
			self.text_emb(text_list),
			self.proms_emb(proms_list),
			self.resps_emb(resps_list),
			sep=self.sep,
		)

		x, m = list_to_tensor(x_list)
		device = x.device

		if state is not None:
			# prefill
			prefill_size = x.shape[1]

			# run the initial prompt to fill the KV cache
			if len(state) == 0:
				for n in range(prefill_size):
					xi = x[:, n, :].unsqueeze(1)
					self.retnet(xi, incremental_state=state, token_embeddings=xi, features_only=True)

			# grab last token(s)
			x = x[:, -1, :].unsqueeze(1)

		if self.arch_type == "transformer":
			x = self.sin_emb.add_pe(x)
			for block in self.blocks:
				x = block(x, m, quant_levels)
		elif self.arch_type == "retnet":
			# to-do: actually make this work and verify it works with recurrent_forward / chunkwise_forward
			x, _ = self.retnet(x, incremental_state=state, token_embeddings=x, features_only=True)

		x = self.classifier(x) * m

		# Remove padding
		h_list = [hi[:li] for hi, li in zip(x, map(len, x_list))]

		if True:
			logits = [hi[:] for hi, li in zip(h_list, map(len, resps_list))]
			ret = [ Categorical(logits=hi / sampling_temperature).sample() for hi in logits ]
			print( [ r for r in ret ] )

		# compute loss if the target is given
		if targ_list is not None:
			if any([l == 0 for l in map(len, targ_list)]):
				raise ValueError("Cannot compute loss given empty targ_list.")

			ignore_sep = torch.tensor(self.ignore_index, device=device)

			# ignore the prompt when computing loss
			prom_list = [
				torch.full_like(t[..., 0], self.ignore_index) for t in proms_list
			]
			# remake input with ignored input prompt
			text_prom_list = self._samplewise_merge_tensors(
				text_list, prom_list, sep=ignore_sep
			)

			for i in range(len(text_prom_list)):
				# ignore computing loss against text/prompt portion of input
				# the NAR doesn't need to compute the loss for it
				if self.resp_loss_only:
					text_prom_list[i][:] = self.ignore_index

				# roll the text/prompt for loss computing
				# the AR benefits from this, for some reason I'll figure out later
				else:
					text_prom_list[i] = text_prom_list[i].roll(-1, dims=0)
					text_prom_list[i][-1] = self.ignore_index

			# for the AR, roll by one and mark the ending with a stop token
			# this coerces the model into properly inferencing causally

			#	 why we don't just append a stop token in the dataloader, who knows
			if shift_targ_list:
				targ_list = [*targ_list] 
				for i in range(len(targ_list)):
					targ_list[i] = targ_list[i].roll(-1, dims=0)
					targ_list[i][-1] = self.stop_token

			# create the new target sequence to compute the loss against
			y_list = self._samplewise_merge_tensors( text_prom_list, targ_list, sep=ignore_sep )

			self.loss = dict(
				nll=F.cross_entropy(
					torch.cat(h_list), # input / predicted logits
					torch.cat(y_list), # target / ground truth
					ignore_index=self.ignore_index,
				)
			)
			self.stats = dict(
				acc = self.accuracy_metric( torch.cat(h_list), torch.cat(y_list) ),
				precision = self.precision_metric( torch.cat(h_list), torch.cat(y_list) ),
			)

		# return the entire generated token string
		if return_all:
			logits = [hi[:] for hi, li in zip(h_list, map(len, resps_list))]
		# return the entire generated response
		elif return_all_resp:
			logits = [hi[-li:] for hi, li in zip(h_list, map(len, resps_list))]
		# return the last chunkwise piece
		elif self.causal and self.recurrent_chunk_size > 0:
			logits = [hi[-self.recurrent_chunk_size:] for hi, li in zip(h_list, map(len, resps_list))]
		# return just the last code
		else:
			logits = [ hi[-1:] for hi in h_list ]
		
		return [ Categorical(logits=hi / sampling_temperature).sample() for hi in logits ]

	def forward(
		self,
		text_list: list[Tensor],
		proms_list: list[Tensor],
		resps_list: list[Tensor] | None = None,
		max_steps: int = 1000,
		sampling_temperature: float = 1.0,
	):
		if resps_list is not None:
			resps_list = [self._interleave(r) for r in resps_list] # guarantees we only have the first levels

			return self._forward(
				text_list=text_list,
				proms_list=proms_list,
				resps_list=self._unsqueeze_list(resps_list),
				targ_list=resps_list,
				quant_levels=None,
				shift_targ_list=True,
				return_all_resp=False,
			)

		device = text_list[0].device
		batch_size = len(text_list)

		resps_list: list[Tensor] = [ torch.zeros(0, device=device).to(torch.int16) for _ in text_list ]
		stopped = torch.zeros(batch_size, device=device).bool()

		state = {} if cfg.inference.recurrent_forward else None

		for n in range(max_steps // max(1, self.recurrent_chunk_size)):
			# get next in sequence

			r = self._forward(
				text_list,
				proms_list,
				self._unsqueeze_list(resps_list),
				sampling_temperature=sampling_temperature,
				state=state
			)

			# append tokens
			for i, ri in enumerate(r):
				if self.stop_token in ri:
					stopped[i] = True
				resps_list[i] = torch.cat([resps_list[i], ri])

			# stop token found
			stopped |= r == self.stop_token
			if stopped.all().item():
				break


		pruned = [self._prune(r) for r in resps_list]
		print( [ r for r in pruned ] )
		deinterleaved = [ self._deinterleave(r) for r in pruned ]
		print( [ r for r in deinterleaved ] )
		return deinterleaved

def example_usage():
	from ..config import cfg
	cfg.trainer.backend = "local"
	cfg.trainer.check_for_oom = False

	from functools import partial

	from einops import repeat

	from ..emb.qnt import decode_to_file
	from ..engines import Engine, Engines
	from tqdm import tqdm, trange

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
		'n_layers': 18,
	}
	models = { "ar": Base(**kwargs).to(device) }

	for name, model in models.items():
		print(f"{name} parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

	engines = Engines({ name: Engine(model=model, optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4)) for name, model in models.items() })

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
	
	def sample( filename, steps=450 * 4 ):
		AR = None

		engines.eval()
		for name, engine in engines.items():
			if name[:2] == "ar":
				AR = engine

		resps_list = AR(text_list, proms_list, max_steps=steps, sampling_temperature=1.0)

		decode_to_file(resps_list[0].cpu(), f"./data/{filename}.wav", device="cpu")
	
	if train:
		sample("init", 15)

		engines.train()
		t = trange(100)
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
