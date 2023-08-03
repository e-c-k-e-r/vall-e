from ..config import cfg
from .base import Base, list_to_tensor, Categorical

import torch

from einops import rearrange
from torch import Tensor
from tqdm import trange

class AR(Base):
	@property
	def n_resp_levels(self) -> int:
		return cfg.models.ar.resp_levels

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
		resp_list: list[Tensor] | None = None,
		max_steps: int = 1000,
		sampling_temperature: float = 1.0,

		naive: bool = True,
	):
		if resp_list is not None:
			return super().forward(
				text_list,
				proms_list,
				self._unsqueeze_list(resp_list),
				resp_list,
				quant_levels=None,
				shift_targ_list=True,
				return_all_resp=False,
			)

		device = text_list[0].device
		resp_list: list[Tensor] = [
			torch.zeros(0, device=device).to(torch.int16) for _ in text_list
		]
		stopped = torch.zeros(len(text_list), device=device).bool()

		if self.arch_type == "transformer":
			naive = True

		chunk_size = 1 # don't really know what to do about this desu

		state = None
		start = 0

		# prefill
		if self.arch_type == "retnet/local":
			# pre-process
			state = [
				[
					torch.zeros(self.retnet.hidden_dim // self.retnet.heads, self.retnet.v_dim // self.retnet.heads, device=device).unsqueeze(0).repeat(len(text_list), 1, 1)
					for _ in range(self.retnet.heads)
				] for _ in range(self.retnet.layers)
			]
			resps_list = self._unsqueeze_list(resp_list)
			x_list = self._samplewise_merge_tensors(
				self.text_emb(text_list),
				self.proms_emb(proms_list),
				self.resps_emb(resps_list),
				sep=self.sep,
			)

			x, m = list_to_tensor(x_list)
			
			start = x.shape[1]

			for i in trange(start-1):
				_, state = self.retnet.forward_recurrent( x[:, i:i+1, :], state, i+1 )

		for n in trange(max_steps // chunk_size):
			# get next in sequence

			r, state = super().forward(
				text_list,
				proms_list,
				self._unsqueeze_list(resp_list),
				sampling_temperature=sampling_temperature,
				state=state,
			)

			# append outputted token
			for i, ri in enumerate(r):
				resp_list[i] = torch.cat([resp_list[i], ri[None]])

			# stop token found
			stopped |= r == self.stop_token
			if stopped.all().item():
				break

		pruned = [self._prune(r) for r in resp_list]
		return pruned


def example_usage():
	from functools import partial

	from einops import repeat

	from ..emb.qnt import decode_to_file
	from ..utils import gather_attribute

	device = "cpu"

	symmap = {'<s>': 1, '</s>': 2, ' ': 3, '.': 4, ',': 5, '!': 6, '?': 7, 'p': 7, 'iː': 8, 'ɚ': 9, 'ˌ': 10, 'dˌ': 11, 'mˌ': 12, 'd': 13, 'ɹ': 14, 'tˈ': 15, 'pˌ': 16, 'uː': 17, 'l': 18, 'æ': 19, 'ɛ': 20, 'ɪ': 21, 'j': 22, 'ʊ': 23, 't': 24, 'n': 25, 'v': 26, 'a': 27, 'o': 28, 'ŋ': 29, 'w': 30, 'ʌ': 31, 'hˈ': 32, 'ɡˈ': 33, 'ə': 34, 'θˈ': 35, 'dˈ': 36, 'wˌ': 37, 'h': 38, 'z': 39, 'k': 40, 'ð': 41, 'ɡˌ': 42, 'ˈ': 43, 'fˈ': 44, 'i': 45, 's': 46, 'ʃ': 47, 'wˈ': 48, 'ðˈ': 49, 'ɹˈ': 50, 'lˈ': 51, 'ɡ': 52, 'oː': 53, 'mˈ': 54, 'e': 55, 'ɑː': 56, 'nˈ': 57, 'm': 58, 'θˌ': 59, 'sˈ': 60, 'f': 61, 'ɔː': 62, 'hˌ': 63, 'b': 64, 'jˈ': 65, 'ɐ': 66, 'ʒˈ': 67, 'θ': 68, 'bˈ': 69, 'ɾ': 70, 'ɜː': 71, 'ʌˈ': 72, 'ʃˌ': 73, 'bˌ': 74, 'kˈ': 75, 'ɔ': 76, 'zˈ': 77, 'ᵻ': 78, 'kˌ': 79, 'vˈ': 80, 'fˌ': 81, 'ʒ': 82, 'ʃˈ': 83, 'ɹˌ': 84, 'tˌ': 85, 'pˈ': 86, 'ðˌ': 87, 'sˌ': 88, 'nˌ': 89, 'lˌ': 90, '̩': 91, 'ʔ': 92, 'vˌ': 93, 'ɪˈ': 94, '"': 95, 'ɪˌ': 96, 'ʒˌ': 97, 'uːˌ': 98, 'ʊˈ': 99, 'jˌ': 100, 'uːˈ': 101, 'iːˈ': 102, 'zˌ': 103, '.ˈ': 104, '…': 105, 'ŋˌ': 106, 'ɐˌ': 107, '—ˈ': 108, 'iˌ': 109, 'iːˌ': 110, 'ɛː': 111, ')': 112, ')ˈ': 113, '(': 114, 'u': 115, '-': 116, 'ɖˈ': 117, 'iˈ': 118, 'ʰˈ': 119, 'ɟˈ': 120, '̃': 121, 'eː': 122, 'ɾˈ': 123, 'r': 124, 'ʰ': 125, '-ˌ': 126, 'ɫ': 127, 'q': 128, '—': 129, 'ʊˌ': 130, 'aː': 131, 'cˈ': 132, '…ˈ': 133, 'c': 134, 'ɳ': 135, 'ɐˈ': 136, 'x': 137, 'ʔˌ': 138, '.ˌ': 139, 'ɑ': 140, '?ˈ': 141, '̩ˈ': 142, '"ˈ': 143, ',ˈ': 144, 'ŋˈ': 145, 'əˌ': 146, '!ˈ': 147, '"ˌ': 148, '?ˌ': 149, ',ˌ': 150, '—ˌ': 151, '̩ˌ': 152, 'əˈ': 153, '!ˌ': 154, 'ɬ': 155, 'ʲ': 156, '¡': 157, 'ɯ': 158, 'qˌ': 159, 'ʑ': 160, 'ʑˈ': 161, '¿': 162, 'ɑːˈ': 163, 'iːː': 164, 'ɛˈ': 165, '¡ˈ': 166, 'æˈ': 167, 'ç': 168, 'ɾˌ': 169, 'ᵻˈ': 170, 'xˈ': 171, 'ɔːˈ': 172, ';': 173, 'ɬˌ': 174, ':': 175, 'ʔˈ': 176, 'ɑːˌ': 177, 'ɬˈ': 178}
	def tokenize(content, lang_marker="en"):
		split = content.split(" ")
		phones = [f"<s>"] + [ " " if not p else p for p in split ] + [f"</s>"]
		return torch.tensor([*map(symmap.get, phones)]).to()

	qnt = torch.load("data/qnt.pt")[0, 0].to(device)
	kwargs = {
		'n_tokens': 1024,
		'd_model': 1024,
		'n_heads': 16,
		'n_layers': 12,
	}

	model = AR(**kwargs).to(device)

	x8 = partial(repeat, pattern="t -> t l", l=2) 
	text_list = [
		#torch.tensor([1, 2, 3], device=device),
		tokenize("ˈ a ɪ   w ɪ l   nˌ ɑː t  ˈ æ s k   ɐ   sˈ ɛ k ə n d   tˈ a ɪ m").to(device),
	]
	proms_list = [
		x8(torch.tensor([1, 2, 3], device=device)),
		#qnt.to(device),
	]
	resp_list = [
		qnt.to(device),
	]

	text_list = text_list[:1]
	proms_list = proms_list[:1]
	resp_list = resp_list[:1]

	model.eval()
	out = model(text_list, proms_list, max_steps=75)[0]
	print("qnt:", qnt.shape, qnt)
	print("out:", out.shape, out)
	wav, sr = decode_to_file(out, "data/test/test.ar.init.wav", device=device)

	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

	model.train()
	for i in trange(60):
		optimizer.zero_grad()
		_ = model(text_list, proms_list, resp_list)

		losses = gather_attribute(model, "loss")
		loss = sum(losses.values())
		loss.backward()
		optimizer.step()

		if i % 20 == 0:
			print(f"iter={i}, {losses}.")
	model.eval()
	out = model(text_list, proms_list, max_steps=400)
	print("qnt:", qnt.shape, qnt)
	for i, o in enumerate(out):
		print("out:", i, o.shape, o)
		wav, sr = decode_to_file(o, f"data/test/test.ar.{i}.wav", device=device)


if __name__ == "__main__":
	example_usage()
