from ..config import cfg
from .base import Base, list_to_tensor, Categorical

import torch
from torch.nn.utils.rnn import pad_sequence

from einops import rearrange
from torch import Tensor
from tqdm import trange

class AR(Base):
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
	def arch_type(self) -> bool:
		return cfg.models.ar.arch_type

	@property
	def n_prom_levels(self) -> int:
		return cfg.models.prom_levels

	@property
	def n_resp_levels(self) -> int:
		return cfg.models.ar.resp_levels

	@property
	def n_max_levels(self) -> int:
		return cfg.models.max_levels

	@property
	def n_tasks(self) -> int:
		return cfg.models.tasks

	@property
	def resp_loss_only(self):
		return False

	def _prune(self, l: Tensor):
		indices = (l == self.stop_token).nonzero()
		if len(indices) == 0:
			return l
		return l[: indices.min().item()]

	@staticmethod
	def _unsqueeze_list(x_list, axis=-1):
		return [x.unsqueeze(dim=axis) for x in x_list]

	def forward(
		self,
		text_list: list[Tensor],
		proms_list: list[Tensor],
		resps_list: list[Tensor] | None = None,
		max_steps: int = 1000,
		sampling_temperature: float = 1.0,

		naive: bool = False,
	):
		if resps_list is not None:
			resps_list = [r[..., 0] for r in resps_list] # guarantees we only have the first levels

			return super().forward(
				text_list=text_list,
				proms_list=proms_list,
				resps_list=self._unsqueeze_list(resps_list),
				targ_list=resps_list,
				quant_levels=None,
				shift_targ_list=True,
				return_all_resp=False,
			)

		device = text_list[0].device
		resps_list: list[Tensor] = [
			torch.zeros(0, device=device).to(torch.int16) for _ in text_list
		]
		stopped = torch.zeros(len(text_list), device=device).bool()

		chunk_size = self.causal_chunk_size # don't really know what to do about this desu

		state = None
		start = 0

		if naive:
			for n in trange(max_steps // max(1, chunk_size)):
				# get next in sequence

				r, state = super().forward(
					text_list,
					proms_list,
					self._unsqueeze_list(resps_list),
					sampling_temperature=sampling_temperature,
					state=state # if not naive else None,
				)

				# append outputted token
				if self.causal_chunk_size > 0:
					for i, ri in enumerate(r):
						resps_list[i] = torch.cat([resps_list[i], ri])
				else:
					for i, ri in enumerate(r):
						resps_list[i] = torch.cat([resps_list[i], ri[None]])


				# stop token found
				stopped |= r == self.stop_token
				if stopped.all().item():
					break
		# to-do: make it work
		# it seems anything that isn't a one-at-a-time sequence does not work, despite generating STOP tokens.
		else:
			resps_list: list[Tensor] = [
				torch.zeros(0, device=device).to(torch.int16) for _ in text_list
			]

			test_list: list[Tensor] = [
				torch.zeros(0, device=device).to(torch.int16) for _ in text_list
			]

			batch_size = len(text_list)

			x_list = self._samplewise_merge_tensors(
				self.text_emb(text_list),
				self.proms_emb(proms_list),
				self.resps_emb(self._unsqueeze_list(resps_list)),
				sep=self.sep,
			)

			x, m = list_to_tensor(x_list)
			device = x.device

			if state is None:
				state = {}

			# pre-fill KV cache
			for n in trange(x.shape[1]):
				xs = x[:, n:(n + 1), :]
				r, _ = self.retnet(xs, incremental_state=state, token_embeddings=xs, features_only=True)
				r = self.classifier(r) * m

				logits = torch.stack([hi[-1] for hi in r])
				r = Categorical(logits=logits / sampling_temperature).sample()

				for i, ri in enumerate(r):
					test_list[i] = torch.cat([test_list[i], ri[None]])

			# append outputted token
			for i, ri in enumerate(r):
				resps_list[i] = torch.cat([resps_list[i], ri[None]])

			start = x.shape[1]
			for n in trange(max_steps // max(1, chunk_size)):
				x_list = self._samplewise_merge_tensors(
					self.text_emb(text_list),
					self.proms_emb(proms_list),
					self.resps_emb(self._unsqueeze_list(resps_list)),
					sep=self.sep,
				)

				x, m = list_to_tensor(x_list)

				xs = x[:, start+n:start+(n+1), :]
				r, _ = self.retnet(xs, incremental_state=state, token_embeddings=xs, features_only=True)
				r = self.classifier(r) * m
				
				logits = torch.stack([hi[-1] for hi in r])
				r = Categorical(logits=logits / sampling_temperature).sample()

				# append outputted token
				for i, ri in enumerate(r):
					resps_list[i] = torch.cat([resps_list[i], ri[None]])

				# stop token found
				stopped |= r == self.stop_token
				if stopped.all().item():
					break

		pruned = [self._prune(r) for r in resps_list]
		return pruned


def example_usage():
	cfg.trainer.backend = "local"
	from functools import partial

	from einops import repeat

	from ..emb.qnt import decode_to_file
	from ..engines import Engine
	from tqdm import tqdm

	device = "cuda"
	x8 = partial(repeat, pattern="t -> t l", l=2) 
	symmap = {'<s>': 1, '</s>': 2, ' ': 3, '.': 4, ',': 5, '!': 6, '?': 7, 'p': 7, 'iː': 8, 'ɚ': 9, 'ˌ': 10, 'dˌ': 11, 'mˌ': 12, 'd': 13, 'ɹ': 14, 'tˈ': 15, 'pˌ': 16, 'uː': 17, 'l': 18, 'æ': 19, 'ɛ': 20, 'ɪ': 21, 'j': 22, 'ʊ': 23, 't': 24, 'n': 25, 'v': 26, 'a': 27, 'o': 28, 'ŋ': 29, 'w': 30, 'ʌ': 31, 'hˈ': 32, 'ɡˈ': 33, 'ə': 34, 'θˈ': 35, 'dˈ': 36, 'wˌ': 37, 'h': 38, 'z': 39, 'k': 40, 'ð': 41, 'ɡˌ': 42, 'ˈ': 43, 'fˈ': 44, 'i': 45, 's': 46, 'ʃ': 47, 'wˈ': 48, 'ðˈ': 49, 'ɹˈ': 50, 'lˈ': 51, 'ɡ': 52, 'oː': 53, 'mˈ': 54, 'e': 55, 'ɑː': 56, 'nˈ': 57, 'm': 58, 'θˌ': 59, 'sˈ': 60, 'f': 61, 'ɔː': 62, 'hˌ': 63, 'b': 64, 'jˈ': 65, 'ɐ': 66, 'ʒˈ': 67, 'θ': 68, 'bˈ': 69, 'ɾ': 70, 'ɜː': 71, 'ʌˈ': 72, 'ʃˌ': 73, 'bˌ': 74, 'kˈ': 75, 'ɔ': 76, 'zˈ': 77, 'ᵻ': 78, 'kˌ': 79, 'vˈ': 80, 'fˌ': 81, 'ʒ': 82, 'ʃˈ': 83, 'ɹˌ': 84, 'tˌ': 85, 'pˈ': 86, 'ðˌ': 87, 'sˌ': 88, 'nˌ': 89, 'lˌ': 90, '̩': 91, 'ʔ': 92, 'vˌ': 93, 'ɪˈ': 94, '"': 95, 'ɪˌ': 96, 'ʒˌ': 97, 'uːˌ': 98, 'ʊˈ': 99, 'jˌ': 100, 'uːˈ': 101, 'iːˈ': 102, 'zˌ': 103, '.ˈ': 104, '…': 105, 'ŋˌ': 106, 'ɐˌ': 107, '—ˈ': 108, 'iˌ': 109, 'iːˌ': 110, 'ɛː': 111, ')': 112, ')ˈ': 113, '(': 114, 'u': 115, '-': 116, 'ɖˈ': 117, 'iˈ': 118, 'ʰˈ': 119, 'ɟˈ': 120, '̃': 121, 'eː': 122, 'ɾˈ': 123, 'r': 124, 'ʰ': 125, '-ˌ': 126, 'ɫ': 127, 'q': 128, '—': 129, 'ʊˌ': 130, 'aː': 131, 'cˈ': 132, '…ˈ': 133, 'c': 134, 'ɳ': 135, 'ɐˈ': 136, 'x': 137, 'ʔˌ': 138, '.ˌ': 139, 'ɑ': 140, '?ˈ': 141, '̩ˈ': 142, '"ˈ': 143, ',ˈ': 144, 'ŋˈ': 145, 'əˌ': 146, '!ˈ': 147, '"ˌ': 148, '?ˌ': 149, ',ˌ': 150, '—ˌ': 151, '̩ˌ': 152, 'əˈ': 153, '!ˌ': 154, 'ɬ': 155, 'ʲ': 156, '¡': 157, 'ɯ': 158, 'qˌ': 159, 'ʑ': 160, 'ʑˈ': 161, '¿': 162, 'ɑːˈ': 163, 'iːː': 164, 'ɛˈ': 165, '¡ˈ': 166, 'æˈ': 167, 'ç': 168, 'ɾˌ': 169, 'ᵻˈ': 170, 'xˈ': 171, 'ɔːˈ': 172, ';': 173, 'ɬˌ': 174, ':': 175, 'ʔˈ': 176, 'ɑːˌ': 177, 'ɬˈ': 178}
	def tokenize(content, lang_marker="en"):
		split = content.split(" ")
		phones = [f"<s>"] + [ " " if not p else p for p in split ] + [f"</s>"]
		return torch.tensor([*map(symmap.get, phones)]).to()

	qnt = torch.load("data/qnt.pt")[0].t()[:, :2].to(device)

	text_list = [
		#torch.tensor([1, 2, 3], device=device),
		tokenize("ˈ a ɪ   w ɪ l   nˌ ɑː t  ˈ æ s k   ɐ   sˈ ɛ k ə n d   tˈ a ɪ m").to(device),
	]
	proms_list = [
		x8(torch.tensor([1, 2, 3], device=device)),
		#qnt.to(device),
	]
	resps_list = [
		qnt.to(device),
	]

	text_list = text_list[:1]
	proms_list = proms_list[:1]
	resps_list = resps_list[:1]

	kwargs = {
		'n_tokens': 1024,
		'd_model': 1024,
		'n_heads': 16,
		'n_layers': 12,
	}
	model = AR(**kwargs).to(device)
	engine = Engine(model=model, optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4))
	
	def sample( name, steps=400 ):
		engine.eval()
		out = engine(text_list, proms_list, max_steps=steps)
		for i, o in enumerate(out):
			wav, sr = decode_to_file(o, f"data/ar.{i}.{name}.wav", device=device)

	def train():
		engine.train()
		t = trange(60)
		for i in t:
			stats = {"step": i}
			stats |= engine.traverse(text_list=text_list, proms_list=proms_list, resps_list=resps_list)

			t.set_description(f"{stats}")

	sample("init", 75)
	train()
	sample("final")

if __name__ == "__main__":
	example_usage()
