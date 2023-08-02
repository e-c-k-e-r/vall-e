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


class Base(nn.Module):
	@property
	def causal(self) -> bool:
		raise NotImplementedError

	@property
	def n_resp_levels(self) -> int:
		raise NotImplementedError

	@property
	def use_stop_token(self) -> bool:
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
	def resp_loss_only(self):
		raise NotImplementedError

	def __init__(
		self,
		n_tokens: int,
		d_model: int = 512,
		n_heads: int = 8,
		n_layers: int = 12,
		p_dropout: float = 0.1,
	):
		super().__init__()
		self.n_tokens = n_tokens
		self.d_model = d_model
		self.n_heads = n_heads
		self.n_layers = n_layers

		causal = self.causal

		# +1 to include the stop token
		n_stop_tokens = 1 if self.use_stop_token else 0
		n_resp_tokens = n_tokens + n_stop_tokens

		self.text_emb = Embedding(n_tokens, d_model)

		# Here I simply use all prom levels
		self.proms_emb = MultiEmbedding(self.n_prom_levels, n_tokens, d_model)
		self.resps_emb = MultiEmbedding(self.n_resp_levels, n_resp_tokens, d_model)

		self.sep = nn.Parameter(torch.randn(d_model))
		
		if self.arch_type == "transformer":
			self.sin_emb = SinusoidalEmbedding(d_model)
			self.blocks = nn.ModuleList([TransformerBlock(
				d_model=d_model,
				n_heads=n_heads,
				p_dropout=p_dropout,
				causal=causal,
				norm_type=self.norm_type,
				n_levels=self.n_resp_levels,
				#tention="retention" if self.use_retnet else "attention"
			) for _ in range(n_layers) ])

		elif self.arch_type == "retnet":
			self.retnet_config = RetNetConfig(
				vocab_size=n_tokens,
				decoder_embed_dim=d_model,
				decoder_retention_heads=n_heads,
				decoder_ffn_embed_dim=d_model * 4,
				decoder_layers=n_layers,
				dropout=p_dropout,
				checkpoint_activations=True,

				chunkwise_recurrent=self.causal,
				recurrent_chunkwise_size=128,
				no_output_layer=True,
				decoder_normalize_before=True,
			)
			self.retnet = RetNetDecoder(
				self.retnet_config
			)
		elif self.arch_type == "retnet/local":
			self.retnet = RetNet(
				layers=n_layers,
				hidden_dim=d_model,
				ffn_size=d_model * 4,
				heads=n_heads,
				dropout=p_dropout,
				norm_type=self.norm_type,
				n_levels=self.n_resp_levels,
				double_v_dim=True
			)
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

	@property
	def stop_token(self):
		if not self.use_stop_token:
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

	def forward(
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

		state: list | None = None,
	):
		"""
		Args:
			text_list: [t] * b
			proms_list: [t' l] * b, l quantization levels.
			resps_list: [t'' l] * b, l quantization levels.
			targ_list: [t''] * b, one quantization level only, when given, loss will be computed
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

		if self.arch_type == "transformer":
			x = self.sin_emb.add_pe(x)
			for block in self.blocks:
				x = block(x, m, quant_levels)
		elif self.arch_type == "retnet":
			x, _ = self.retnet(x, incremental_state=state, token_embeddings=x, features_only=True)
			state = self.retnet.get_incremental_state( state, 'prev_state' )
		elif self.arch_type == "retnet/local":
			# recurrent inferencing
			if self.causal and state is not None:
				last = x.shape[1]
				x, state = self.retnet.forward_recurrent(
					x[:, last-1:last, :], # nasty way to grab the last embedding to forward
					state,
					last
				)
			else:
				x = self.retnet( x, quant_levels )
			
		x = self.classifier(x) * m

		# Remove padding
		h_list = [hi[:li] for hi, li in zip(x, map(len, x_list))]


		# compute loss if the target is given
		if targ_list is not None:
			if any([l == 0 for l in map(len, targ_list)]):
				raise ValueError("Cannot compute loss given empty targ_list.")

			ignore_sep = torch.tensor(self.ignore_index, device=x.device)

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
				# the AR benefits from this
				else:
					text_prom_list[i] = text_prom_list[i].roll(-1, dims=0)
					text_prom_list[i][-1] = self.ignore_index

			# necessary to roll the target if recurrently/causally/autoregressively generating, or it won't be able to work
			if shift_targ_list:
				targ_list = [*targ_list] 
				for i in range(len(targ_list)):
					targ_list[i] = targ_list[i].roll(-1, dims=0)
					targ_list[i][-1] = self.stop_token

			# generate the sequence
			y_list = self._samplewise_merge_tensors( text_prom_list, targ_list, sep=ignore_sep )

			self.loss = dict(
				nll=F.cross_entropy(
					torch.cat(h_list), # input / predicted logits
					torch.cat(y_list), # target / ground truth
					ignore_index=self.ignore_index,
				)
			)
			self.loss['acc'] = self.accuracy_metric( torch.cat(h_list), torch.cat(y_list) )
			self.loss['precision'] = self.precision_metric( torch.cat(h_list), torch.cat(y_list) )

			del targ_list
			del prom_list
			del text_prom_list
			del y_list
		
		# return the entire generated token string
		if return_all:
			logits = [hi[:] for hi, li in zip(h_list, map(len, resps_list))]
			ret = [Categorical(logits=hi / sampling_temperature).sample() for hi in logits]
		# return the entire generated response
		elif return_all_resp:
			logits = [hi[-li:] for hi, li in zip(h_list, map(len, resps_list))]
			ret = [ Categorical(logits=hi / sampling_temperature).sample() for hi in logits ]
		# return just the last code
		else:
			logits = torch.stack([hi[-1] for hi in h_list])
			ret = Categorical(logits=logits / sampling_temperature).sample()

		del x_list
		del h_list
		
		return ret, state

def example_usage():
	from functools import partial

	from einops import repeat
	from tqdm import trange
	
	from ..utils import gather_attribute
	from ..emb.qnt import decode_to_file
	from .ar import AR
	from .nar import NAR

	symmap = {'<s>': 1, '</s>': 2, ' ': 3, '.': 4, ',': 5, '!': 6, '?': 7, 'p': 7, 'iː': 8, 'ɚ': 9, 'ˌ': 10, 'dˌ': 11, 'mˌ': 12, 'd': 13, 'ɹ': 14, 'tˈ': 15, 'pˌ': 16, 'uː': 17, 'l': 18, 'æ': 19, 'ɛ': 20, 'ɪ': 21, 'j': 22, 'ʊ': 23, 't': 24, 'n': 25, 'v': 26, 'a': 27, 'o': 28, 'ŋ': 29, 'w': 30, 'ʌ': 31, 'hˈ': 32, 'ɡˈ': 33, 'ə': 34, 'θˈ': 35, 'dˈ': 36, 'wˌ': 37, 'h': 38, 'z': 39, 'k': 40, 'ð': 41, 'ɡˌ': 42, 'ˈ': 43, 'fˈ': 44, 'i': 45, 's': 46, 'ʃ': 47, 'wˈ': 48, 'ðˈ': 49, 'ɹˈ': 50, 'lˈ': 51, 'ɡ': 52, 'oː': 53, 'mˈ': 54, 'e': 55, 'ɑː': 56, 'nˈ': 57, 'm': 58, 'θˌ': 59, 'sˈ': 60, 'f': 61, 'ɔː': 62, 'hˌ': 63, 'b': 64, 'jˈ': 65, 'ɐ': 66, 'ʒˈ': 67, 'θ': 68, 'bˈ': 69, 'ɾ': 70, 'ɜː': 71, 'ʌˈ': 72, 'ʃˌ': 73, 'bˌ': 74, 'kˈ': 75, 'ɔ': 76, 'zˈ': 77, 'ᵻ': 78, 'kˌ': 79, 'vˈ': 80, 'fˌ': 81, 'ʒ': 82, 'ʃˈ': 83, 'ɹˌ': 84, 'tˌ': 85, 'pˈ': 86, 'ðˌ': 87, 'sˌ': 88, 'nˌ': 89, 'lˌ': 90, '̩': 91, 'ʔ': 92, 'vˌ': 93, 'ɪˈ': 94, '"': 95, 'ɪˌ': 96, 'ʒˌ': 97, 'uːˌ': 98, 'ʊˈ': 99, 'jˌ': 100, 'uːˈ': 101, 'iːˈ': 102, 'zˌ': 103, '.ˈ': 104, '…': 105, 'ŋˌ': 106, 'ɐˌ': 107, '—ˈ': 108, 'iˌ': 109, 'iːˌ': 110, 'ɛː': 111, ')': 112, ')ˈ': 113, '(': 114, 'u': 115, '-': 116, 'ɖˈ': 117, 'iˈ': 118, 'ʰˈ': 119, 'ɟˈ': 120, '̃': 121, 'eː': 122, 'ɾˈ': 123, 'r': 124, 'ʰ': 125, '-ˌ': 126, 'ɫ': 127, 'q': 128, '—': 129, 'ʊˌ': 130, 'aː': 131, 'cˈ': 132, '…ˈ': 133, 'c': 134, 'ɳ': 135, 'ɐˈ': 136, 'x': 137, 'ʔˌ': 138, '.ˌ': 139, 'ɑ': 140, '?ˈ': 141, '̩ˈ': 142, '"ˈ': 143, ',ˈ': 144, 'ŋˈ': 145, 'əˌ': 146, '!ˈ': 147, '"ˌ': 148, '?ˌ': 149, ',ˌ': 150, '—ˌ': 151, '̩ˌ': 152, 'əˈ': 153, '!ˌ': 154, 'ɬ': 155, 'ʲ': 156, '¡': 157, 'ɯ': 158, 'qˌ': 159, 'ʑ': 160, 'ʑˈ': 161, '¿': 162, 'ɑːˈ': 163, 'iːː': 164, 'ɛˈ': 165, '¡ˈ': 166, 'æˈ': 167, 'ç': 168, 'ɾˌ': 169, 'ᵻˈ': 170, 'xˈ': 171, 'ɔːˈ': 172, ';': 173, 'ɬˌ': 174, ':': 175, 'ʔˈ': 176, 'ɑːˌ': 177, 'ɬˈ': 178}
	def tokenize(content, lang_marker="en"):
		split = content.split(" ")
		phones = [f"<s>"] + [ " " if not p else p for p in split ] + [f"</s>"]
		return torch.tensor([*map(symmap.get, phones)]).to()

	device = "cpu"

	kwargs = {
		'n_tokens': 1024,
		'd_model': 1024,
		'n_heads': 16,
		'n_layers': 12,
	}
	model_ar = AR(**kwargs).to(device)
	model_nar = NAR(**kwargs).to(device)

	train = True

	if train:
		qnt = torch.load("data/qnt.pt").to(device)
		text_list = [
			tokenize("ˈ a ɪ   w ɪ l   nˌ ɑː t  ˈ æ s k   ɐ   sˈ ɛ k ə n d   tˈ a ɪ m").to(device),
			#tokenize("ˌ ɔ n   ɡˌ o ʊ ɪ ŋ   hˈ o ʊ m   ð ə   tˈ uː   f ɹˈ ɛ n d z   fˈ a ʊ n d   ɐ   lˈ ɛ ɾ ɚ   f ɹ ʌ m  ˈ æ θ o ʊ z ,   hˌ uː   d ɪ zˈ a ɪ ɚ d   ðˌ ɛ m   t ə   mˈ iː t   hˌ ɪ m   æ t   ð ə   ɡ ɹˈ æ n d   t ʃˈ ɑː ɹ l ɪ mˌ æ ɡ n i   ɔ n ð ə   fˈ ɑː l o ʊ ɪ ŋ   dˈ e ɪ .").to(device),
		]

		x8 = partial(repeat, pattern="t -> t l", l=2) 
		proms_list = [
			qnt[0][:2,:].t().to(device),
			#x8(torch.tensor([1, 2, 3], device=device)),
			# x8(torch.tensor([2, 3], device=device)),
		]

		resp_list_ar = [
			qnt[0,0].to(device),
			# qnt[0,0].to(device),
		]

		resp_list_nar = [
			qnt[0][:2,:].t().to(device),
			# qnt[0][:2,:].t().to(device),
		]

		model_ar.train()
		optimizer = torch.optim.AdamW(model_ar.parameters(), lr=1e-4)
		for i in trange(60):
			optimizer.zero_grad()
			_ = model_ar(text_list, proms_list, resp_list_ar)

			losses = gather_attribute(model_ar, "loss")
			loss = sum(losses.values())
			loss.backward()
			optimizer.step()

			if i % 20 == 0:
				print(f"iter={i}, {losses}.")

		model_nar.train()
		optimizer = torch.optim.AdamW(model_nar.parameters(), lr=1e-4)
		for i in trange(60):
			optimizer.zero_grad()

			_ = model_nar(text_list, proms_list, resps_list=resp_list_nar)

			losses = gather_attribute(model_nar, "loss")
			loss = sum(losses.values())
			loss.backward()
			optimizer.step()

			if i % 20 == 0:
				stats = {k: v.item() for k, v in losses.items()}
				stats["loss"] = loss.item()
				print(f"iter={i}, {stats}.")
	else:
		qnt = torch.load("data/test/test.qnt.pt")[0][:2,:].t().to(device)
		text_list = [
			#tokenize("ˈ a ɪ   w ɪ l   nˌ ɑː t  ˈ æ s k   ɐ   sˈ ɛ k ə n d   tˈ a ɪ m").to(device),
			tokenize("ˌ ɔ n   ɡˌ o ʊ ɪ ŋ   hˈ o ʊ m   ð ə   tˈ uː   f ɹˈ ɛ n d z   fˈ a ʊ n d   ɐ   lˈ ɛ ɾ ɚ   f ɹ ʌ m  ˈ æ θ o ʊ z ,   hˌ uː   d ɪ zˈ a ɪ ɚ d   ðˌ ɛ m   t ə   mˈ iː t   hˌ ɪ m   æ t   ð ə   ɡ ɹˈ æ n d   t ʃˈ ɑː ɹ l ɪ mˌ æ ɡ n i   ɔ n ð ə   fˈ ɑː l o ʊ ɪ ŋ   dˈ e ɪ .").to(device),
		]
		proms_list = [
			qnt.to(device),
		]
		model_ar.load_state_dict(torch.load("data/test/ar.pth"))
		model_nar.load_state_dict(torch.load("data/test/nar.pth"))        

	model_ar.eval()
	resp_list = model_ar(text_list, proms_list, max_steps=300, sampling_temperature=1.0)
	resps_list = [r.unsqueeze(-1) for r in resp_list]
	
	print("qnt:", qnt.shape, qnt)
	print("out:", resp_list[0].shape, resp_list[0])
	wav, sr = decode_to_file(resp_list[0], "data/test/test.ar.init.wav", device=device)
	print(wav, sr)

	model_nar.eval()
	codes = model_nar(
		text_list,
		proms_list,
		resps_list=resps_list,
		sampling_temperature=1.0,
	)[0]


	print("qnt:", qnt.shape, qnt)
	print("codes:", codes.shape, codes)

	wav, sr = decode_to_file(codes, "data/test/test.ar+nar.init.wav", device=device)
	print(wav, sr)

if __name__ == "__main__":
	example_usage()
