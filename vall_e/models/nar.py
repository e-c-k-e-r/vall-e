from ..config import cfg
from .base import Base

import torch

from torch import Tensor
from tqdm import trange

class NAR(Base):
	@property
	def causal(self):
		return False

	@property
	def arch_type(self) -> str:
		if hasattr(self, "config") and self.config:
			return self.config.arch_type
		return cfg.models.nar.arch_type

	@property
	def norm_type(self):
		return "ln" if self.n_resp_levels == 1 else "adaln"

	@property
	def n_prom_levels(self) -> int:
		return cfg.models.prom_levels

	@property
	def n_resp_levels(self) -> int:
		if hasattr(self, "config") and self.config:
			return self.config.resp_levels
		return cfg.models.nar.resp_levels

	@property
	def n_max_levels(self) -> int:
		return cfg.models.max_levels

	@property
	def n_tasks(self) -> int:
		return cfg.models.nar.tasks

	@property
	def n_langs(self) -> int:
		return cfg.models.nar.langs

	@property
	def version(self) -> int:
		if hasattr(self, "config") and self.config:
			return self.config.version
		return cfg.models.nar.version

	@property
	def recurrent_chunk_size(self) -> int:
		return 0

	"""
	@property
	def rotary_embedding_base(self) -> float:
		if hasattr(self, "config") and self.config:
			return self.config.rotary_embedding_base
		return cfg.models.nar.rotary_embedding_base
	"""

	@property
	def interleave(self) -> bool:
		return False

	@property
	def monolithic(self) -> bool:
		return False

	def forward(
		self,
		text_list: list[Tensor],
		proms_list: list[Tensor],
		resps_list: list[Tensor],
		lang_list: list[Tensor] | None = None,
		max_levels: int = 0,
		sampling_temperature: float = 0.2,
		sampling_min_temperature: float = -1.0,
		sampling_top_k: int = -100,
		sampling_top_p: float = 1.0,
		sampling_repetition_penalty: float = 1.0,
		sampling_repetition_penalty_decay: float = 0.0,
		sampling_length_penalty: float = 0.0, # unused
		sampling_beam_width: int = 0, # unused
		sampling_mirostat_tau: float = 0.0, # unused
	):
		"""
		Args:
			text_list: [t] * b
			proms_list: [t' l] * b, l=8
			resps_list: [t'' l] * b, l=1 or 8, 1 for testing and 8 for training.
		Returns:
			[t'' l], l=8 if testing. empty list will be returned during training.
		"""

		n_levels_set = {r.shape[-1] for r in resps_list}

		if len(n_levels_set) > 1:
			raise ValueError(f"Please give only one level, got {n_levels_set}.")

		n_levels = next(iter(n_levels_set))

		device = text_list[0].device

		if n_levels == self.n_resp_levels + 1:
			assert resps_list is not None

			quant_levels = torch.randint(0, self.n_resp_levels, (len(resps_list),))

			prev_list = [o[..., : l + 1] for o, l in zip(resps_list, quant_levels)]
			targ_list = [o[..., l + 1] for o, l in zip(resps_list, quant_levels)]

			#quant_levels = quant_levels.to(device=device)

			logits = super().forward(
				text_list=text_list,
				proms_list=proms_list,
				resps_list=prev_list,
				targ_list=targ_list,
				lang_list=lang_list,
				quant_levels=quant_levels,
			)

			prev_list = []
		else:
			prev_list = resps_list
			if max_levels == 0:
				max_levels = self.n_resp_levels

			while True:
				level = prev_list[0].shape[-1] - 1

				if level >= max_levels: # min(max_levels, self.n_resp_levels): # commented out to experiment with exceeding trained levels
					break

				quant_levels = torch.full((len(text_list),), level, device=device)

				logits = super().forward(
					text_list=text_list,
					proms_list=proms_list,
					resps_list=prev_list,
					lang_list=lang_list,
					quant_levels=quant_levels,
				)

				resps_list = super().sample(
					logits=logits,
					resps_list=prev_list,
					quant_levels=quant_levels,

					temperature=sampling_temperature,
					min_temperature=sampling_min_temperature,
					top_p=sampling_top_p,
					top_k=sampling_top_k,
					repetition_penalty=sampling_repetition_penalty,
					repetition_penalty_decay=sampling_repetition_penalty_decay,
					#length_penalty=sampling_length_penalty,
					#beam_width=sampling_beam_width,
					#mirostat_tau=sampling_mirostat_tau,
					#mirostat_state=mirostat_state,
				)

				prev_list = [ torch.cat([rs, r.unsqueeze(-1).to(device)], dim=-1) for rs, r in zip(prev_list, resps_list) ]

		return prev_list

def example_usage():
	cfg.trainer.backend = "local"
	from functools import partial

	from einops import repeat

	from ..emb.qnt import decode_to_file
	from ..engines import Engine
	from tqdm import tqdm
	from ..utils import wrapper as ml

	device = "cuda"
	x8 = partial(repeat, pattern="t -> t l", l=cfg.models.prom_levels)
	symmap = {'<s>': 1, '</s>': 2, ' ': 3, '.': 4, ',': 5, '!': 6, '?': 7, 'p': 7, 'iː': 8, 'ɚ': 9, 'ˌ': 10, 'dˌ': 11, 'mˌ': 12, 'd': 13, 'ɹ': 14, 'tˈ': 15, 'pˌ': 16, 'uː': 17, 'l': 18, 'æ': 19, 'ɛ': 20, 'ɪ': 21, 'j': 22, 'ʊ': 23, 't': 24, 'n': 25, 'v': 26, 'a': 27, 'o': 28, 'ŋ': 29, 'w': 30, 'ʌ': 31, 'hˈ': 32, 'ɡˈ': 33, 'ə': 34, 'θˈ': 35, 'dˈ': 36, 'wˌ': 37, 'h': 38, 'z': 39, 'k': 40, 'ð': 41, 'ɡˌ': 42, 'ˈ': 43, 'fˈ': 44, 'i': 45, 's': 46, 'ʃ': 47, 'wˈ': 48, 'ðˈ': 49, 'ɹˈ': 50, 'lˈ': 51, 'ɡ': 52, 'oː': 53, 'mˈ': 54, 'e': 55, 'ɑː': 56, 'nˈ': 57, 'm': 58, 'θˌ': 59, 'sˈ': 60, 'f': 61, 'ɔː': 62, 'hˌ': 63, 'b': 64, 'jˈ': 65, 'ɐ': 66, 'ʒˈ': 67, 'θ': 68, 'bˈ': 69, 'ɾ': 70, 'ɜː': 71, 'ʌˈ': 72, 'ʃˌ': 73, 'bˌ': 74, 'kˈ': 75, 'ɔ': 76, 'zˈ': 77, 'ᵻ': 78, 'kˌ': 79, 'vˈ': 80, 'fˌ': 81, 'ʒ': 82, 'ʃˈ': 83, 'ɹˌ': 84, 'tˌ': 85, 'pˈ': 86, 'ðˌ': 87, 'sˌ': 88, 'nˌ': 89, 'lˌ': 90, '̩': 91, 'ʔ': 92, 'vˌ': 93, 'ɪˈ': 94, '"': 95, 'ɪˌ': 96, 'ʒˌ': 97, 'uːˌ': 98, 'ʊˈ': 99, 'jˌ': 100, 'uːˈ': 101, 'iːˈ': 102, 'zˌ': 103, '.ˈ': 104, '…': 105, 'ŋˌ': 106, 'ɐˌ': 107, '—ˈ': 108, 'iˌ': 109, 'iːˌ': 110, 'ɛː': 111, ')': 112, ')ˈ': 113, '(': 114, 'u': 115, '-': 116, 'ɖˈ': 117, 'iˈ': 118, 'ʰˈ': 119, 'ɟˈ': 120, '̃': 121, 'eː': 122, 'ɾˈ': 123, 'r': 124, 'ʰ': 125, '-ˌ': 126, 'ɫ': 127, 'q': 128, '—': 129, 'ʊˌ': 130, 'aː': 131, 'cˈ': 132, '…ˈ': 133, 'c': 134, 'ɳ': 135, 'ɐˈ': 136, 'x': 137, 'ʔˌ': 138, '.ˌ': 139, 'ɑ': 140, '?ˈ': 141, '̩ˈ': 142, '"ˈ': 143, ',ˈ': 144, 'ŋˈ': 145, 'əˌ': 146, '!ˈ': 147, '"ˌ': 148, '?ˌ': 149, ',ˌ': 150, '—ˌ': 151, '̩ˌ': 152, 'əˈ': 153, '!ˌ': 154, 'ɬ': 155, 'ʲ': 156, '¡': 157, 'ɯ': 158, 'qˌ': 159, 'ʑ': 160, 'ʑˈ': 161, '¿': 162, 'ɑːˈ': 163, 'iːː': 164, 'ɛˈ': 165, '¡ˈ': 166, 'æˈ': 167, 'ç': 168, 'ɾˌ': 169, 'ᵻˈ': 170, 'xˈ': 171, 'ɔːˈ': 172, ';': 173, 'ɬˌ': 174, ':': 175, 'ʔˈ': 176, 'ɑːˌ': 177, 'ɬˈ': 178}
	def tokenize(content, lang_marker="en"):
		split = content.split(" ")
		phones = [f"<s>"] + [ " " if not p else p for p in split ] + [f"</s>"]
		return torch.tensor([*map(symmap.get, phones)]).to()

	# to-do: unmangle this and the resp shit
	qnt = torch.load("data/qnt.pt")[0].t()[:, :cfg.models.prom_levels].to(device)

	text_list = [
		#torch.tensor([1, 2, 3], device=device),
		tokenize("ˈ a ɪ   w ɪ l   nˌ ɑː t  ˈ æ s k   ɐ   sˈ ɛ k ə n d   tˈ a ɪ m").to(device),
	]

	proms_list = [
		x8(torch.tensor([2, 3], device=device)),
	]

	resps_list = [
		qnt.to(device),
	]

	kwargs = {
		'n_tokens': 1024,
		'd_model': 1024,
		'n_heads': 16,
		'n_layers': 12,
	}
	model = NAR(**kwargs).to(device)
	steps = 500
	optimizer = ml.Prodigy(model.parameters(), lr=1.0)
	engine = Engine(model=model, optimizer=optimizer)

	def sample( name ):
		engine.eval()
		codes = engine( text_list, proms_list, resps_list=[r[..., 0].unsqueeze(-1) for r in resps_list], sampling_temperature=0.2 )
		decode_to_file( codes[0], f"data/nar.{name}.wav", device )

	def train():
		engine.train()
		t = trange(steps)
		for i in t:
			stats = {"step": i}
			stats |= engine.traverse(text_list=text_list, proms_list=proms_list, resps_list=resps_list)

			tqdm.write(f"{stats}")

	sample("init")
	train()
	sample("final")


if __name__ == "__main__":
	example_usage()
