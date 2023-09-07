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
"""
class PromEmbedding(nn.Module):
	def __init__(self, n_levels, n_tokens, token_dim):
		super().__init__()
		self.n_levels = n_levels
		self.embeddings = nn.ModuleList([nn.Embedding(n_tokens, token_dim) for _ in range(self.n_levels)])
	
	def forward(self, x_list: list[Tensor] ) -> list[Tensor]:
		if len(x_list) == 0:
			return []

		return [ sum([ self.embeddings[k](xi[:, k]) for k in range(xi.shape[-1]) ]) for i, xi in enumerate(x_list) ]

class RespEmbedding(nn.Module):
	def __init__(self, n_levels, n_tokens, token_dim):
		super().__init__()
		self.n_levels = n_levels
		self.embeddings = nn.ModuleList([nn.Embedding(n_tokens, token_dim) for _ in range(self.n_levels)])
	
	def forward(self, x_list: list[Tensor], quant_levels: Tensor | None = None) -> list[Tensor]:
		if len(x_list) == 0:
			return []
		res = [ self.embeddings[quant_levels[i] if quant_levels is not None else 0](xi) for i, xi in enumerate(x_list) ]
		return res
"""
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
	def n_tasks(self) -> int:
		raise NotImplementedError

	@property
	def recurrent_chunk_size(self) -> int:
		raise NotImplementedError
	
	@property
	def interleave(self) -> bool:
		return False

	@property
	def dual(self) -> bool:
		return False

	@property
	def n_embeddings(self):
		return self.n_resp_levels if self.dual else 1

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
		n_prom_tokens = n_tokens + (self.n_tasks - 1) # - 1 because tts is an inherent task
		n_resp_tokens = n_tokens + (1 if self.causal else 0) # AR requires a stop token to... know when to stop

		self.text_emb = Embedding(n_tokens, d_model)

		self.proms_emb = MultiEmbedding(self.n_prom_levels, n_prom_tokens, d_model)
		if self.n_embeddings == 1:
			self.resps_emb = MultiEmbedding(self.n_resp_levels, n_resp_tokens, d_model)
		else:
			self.resps_emb = nn.ModuleList([ MultiEmbedding(self.n_resp_levels, n_resp_tokens, d_model) for _ in range(self.n_embeddings) ])
		"""
		if self.n_embeddings == 1:
			self.resps_emb = MultiEmbedding(self.n_resp_levels, n_resp_tokens, d_model)
		else:
			self.resps_emb = RespEmbedding(self.n_resp_levels, n_resp_tokens, d_model)
		"""

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
				decoder_retention_heads=n_heads,
				decoder_ffn_embed_dim=d_model * 4,
				decoder_layers=n_layers,
				dropout=p_dropout,
				checkpoint_activations=self.activation_checkpointing,

				chunkwise_recurrent=self.causal and self.recurrent_chunk_size > 0,
				recurrent_chunkwise_size=self.recurrent_chunk_size if self.causal else 0,
				no_output_layer=True,
				decoder_normalize_before=True,
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

	@overload
	def forward(
		self,
		text_list: list[Tensor],
		proms_list: list[Tensor],
		resps_list: list[Tensor],
		targ_list: list[Tensor] | None = None,
		quant_levels: Tensor | None = None,
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
		sampling_temperature: float = 1.0,
	) -> list[Tensor]:
		...

	def forward(
		self,
		text_list: list[Tensor],
		proms_list: list[Tensor],
		resps_list: list[Tensor],
		targ_list: list[Tensor] | None = None,

		quant_levels: Tensor | None = None,
		sampling_temperature: float = 1.0,

		state: dict | None = None,
	):
		if self.n_embeddings == 1:
			x_list = self._samplewise_merge_tensors(
				self.text_emb(text_list),
				self.proms_emb(proms_list),
				self.resps_emb(resps_list),
				sep=self.sep,
			)
		else:
			x_list = self._samplewise_merge_tensors(
				self.text_emb(text_list),
				self.proms_emb(proms_list),
				self.resps_emb[0 if quant_levels is None else 1](resps_list),
				#self.resps_emb(resps_list, quant_levels),
				sep=self.sep,
			)

		x, m = list_to_tensor(x_list)
		
		batch_size = len(text_list)
		device = x.device

		if state is not None:
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
			x = self.sin_emb.add_pe(x)
			l = torch.zeros((batch_size,), dtype=torch.int32) if quant_levels is None else quant_levels
			l = l.to(device)
			for block in self.blocks:
				x = block(x, m, l)

		elif self.arch_type == "retnet":
			x, _ = self.retnet(x, incremental_state=state, token_embeddings=x, features_only=True)
			
		x = self.classifier(x) * m

		# Remove padding
		h_list = [hi[:li] for hi, li in zip(x, map(len, x_list))]

		# compute loss if the target is given
		if targ_list is not None:
			if any([l == 0 for l in map(len, targ_list)]):
				raise ValueError("Cannot compute loss given empty targ_list.")

			ignore_sep = torch.tensor(self.ignore_index, device=device)

			# create a tensor sequence with one RVQ-bin of the input prompt, but with `ignore_index`, as the prompt is not neeeded for computing the loss against
			prom_list = [ torch.full_like(t[..., 0], self.ignore_index) for t in proms_list ]
			# remake input sequence
			text_prom_list = self._samplewise_merge_tensors( text_list, prom_list, sep=ignore_sep )

			# process each batch
			for i in range(len(text_prom_list)):
				# for the AR, shift the text/input prompt into the future by 1, and ignore the rolled back text token
				if quant_levels is None:
					text_prom_list[i] = text_prom_list[i].roll(-1, dims=0)
					text_prom_list[i][-1] = self.ignore_index
				# for the NAR, ignore completely computing the loss against the text prompt
				else:
					text_prom_list[i][:] = self.ignore_index

			# adjust the target sequence if needed for the AR
			if quant_levels is None:
				# creates a copy because this is aliased against input response sequence
				targ_list = [*targ_list] 
				# shift the target response into the future by 1, and mark the rolled back token / last token as a stop token
				# this prepares the AR to actually generate autoregressive sequences
				for i in range(len(targ_list)):
					targ_list[i] = targ_list[i].roll(-1, dims=0)
					targ_list[i][-1] = self.stop_token

			# create the new target sequence to compute the loss against
			y_list = self._samplewise_merge_tensors( text_prom_list, targ_list, sep=ignore_sep )

			self.loss = dict(
				# "nll" was in the original implementation and should actually just be called something else
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
		return_all = False
		if return_all:
			logits = [hi[:] for hi, li in zip(h_list, map(len, resps_list))]
		# return the entire generated response
		elif quant_levels is not None:
			logits = [hi[-li:] for hi, li in zip(h_list, map(len, resps_list))]
		# return the last chunkwise piece
		elif self.causal and self.recurrent_chunk_size > 0:
			logits = [hi[-self.recurrent_chunk_size:] for hi, li in zip(h_list, map(len, resps_list))]
		# return just the last code
		else:
			logits = [ hi[-1:] for hi in h_list ]
		
		return [ Categorical(logits=hi / sampling_temperature).sample() for hi in logits ]

def example_usage():
	from ..config import cfg
	cfg.trainer.backend = "local"
	cfg.trainer.check_for_oom = False

	from functools import partial

	from einops import repeat

	from ..emb.qnt import decode_to_file
	from ..engines import Engine, Engines
	from tqdm import tqdm, trange

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
	
	def sample( name, steps=400 ):
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
		t = trange(60)
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
