from .base import Base, list_to_tensor, Categorical
from ..config import cfg

import torch
from torch.nn.utils.rnn import pad_sequence

import random
import math
from einops import rearrange
from torch import Tensor
from tqdm import trange

from ..emb.qnt import trim

class AR_NAR(Base):
	@property
	def causal(self):
		return True

	@property
	def norm_type(self):
		return "ln" # if self.n_resp_levels == 1 else "adaln"

	@property
	def arch_type(self) -> str:
		if hasattr(self, "config") and self.config:
			return self.config.arch_type
		return cfg.models.ar_nar.arch_type

	@property
	def n_prom_levels(self) -> int:
		return cfg.models.prom_levels

	@property
	def n_resp_levels(self) -> int:
		if hasattr(self, "config") and self.config:
			return self.config.resp_levels
		return cfg.models.ar_nar.resp_levels

	@property
	def n_max_levels(self) -> int:
		return cfg.models.max_levels

	@property
	def n_tasks(self) -> int:
		return cfg.models.tasks

	@property
	def recurrent_chunk_size(self) -> int:
		return 0

	"""
	@property
	def rotary_embedding_base(self) -> float:
		if hasattr(self, "config") and self.config:
			return self.config.rotary_embedding_base
		return cfg.models.ar_nar.rotary_embedding_base
	"""

	@property
	def interleave(self) -> bool:
		return False
	
	@property
	def monolithic(self) -> bool:
		return True

	@property
	def version(self) -> int:
		if hasattr(self, "config") and self.config:
			return self.config.version
		return cfg.models.ar_nar.version

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
		max_levels: int = 7,
		sampling_temperature: float = 1.0,
		sampling_min_temperature: float = -1.0,
		sampling_top_k: int = -100,
		sampling_top_p: float = 1.0,
		sampling_repetition_penalty: float = 1.0,
		sampling_repetition_penalty_decay: float = 0.0,
		sampling_length_penalty: float = 0.0,
		sampling_beam_width: int = 0,
		sampling_mirostat_tau: float = 0.0,
		sampling_mirostat_eta: float = 0.1,
	):
		device = text_list[0].device
		batch_size = len(text_list)

		# is training or NAR
		if resps_list is not None:
			n_levels_set = {r.shape[-1] for r in resps_list}
			n_levels = next(iter(n_levels_set))

			# is training
			if n_levels == self.n_resp_levels:
				if cfg.models.ar_nar.p_ar_level == "auto" or cfg.models.ar_nar.p_ar_level is None:
					quant_levels = torch.randint(0, self.n_resp_levels, (batch_size,)) # randomly select a target RVQ-bin level (0 being AR, 1+ being NAR)
				else:
					quant_levels = torch.Tensor([ [ 0 if random.random() < cfg.models.ar_nar.p_ar_level else random.randint(1, self.n_resp_levels) ] for _ in range(batch_size) ])

				targ_list = [r[..., l] for r, l in zip(resps_list, quant_levels)] # ensures we only have 1 RVQ-bin (our target)
				resps_list = [r if l == 0 else r[..., :l] for r, l in zip(resps_list, quant_levels)] # yes I can just do min(1, l)
				
				if cfg.experimental:
					proms_list = [ r if l == 0 else trim(r, 75 * 3) for r, l in zip(proms_list, quant_levels) ] # trim input prompt to 3 seconds
				
				#quant_levels.to(device=device)

				return super().forward(
					text_list=text_list,
					proms_list=proms_list,
					resps_list=resps_list,
					targ_list=targ_list,
					quant_levels=quant_levels,
				)
			# is NAR
			prev_list = resps_list
			if max_levels == 0:
				max_levels = self.n_resp_levels

			while True:
				level = prev_list[0].shape[-1]

				if level >= max_levels + 1: # min(max_levels + 1, self.n_resp_levels): # commented out to experiment with exceeding trained levels
					break

				quant_levels = torch.full((len(text_list),), level, device=device)

				logits = super().forward(
					text_list=text_list,
					proms_list=proms_list,
					resps_list=prev_list,
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
					#mirostat=mirostat,
				)

				prev_list = [ torch.cat([rs, r.unsqueeze(-1)], dim=-1) for rs, r in zip(prev_list, resps_list) ]

			return prev_list

		# is AR
		sequence_list = [ torch.zeros(0, device=device).to(torch.int16) for _ in text_list ]
		stopped = torch.zeros(batch_size, device=device).bool()

		recurrent_state = {} if cfg.inference.recurrent_forward else None
		mirostat = [
			{"n": 1024, "tau": sampling_mirostat_tau, "eta": sampling_mirostat_eta, "max_surprise": sampling_mirostat_eta * 2, "error_surprise": 0, "running_total_surprise": 0}
		] * batch_size if sampling_mirostat_tau > 0.0 else None

		sampling_beam_width_use_logs = True
		scores = [ 1.0 ] * sampling_beam_width

		if self.interleave:
			max_steps *= self.n_prom_levels

		# get next in sequence
		for n in trange(max_steps // max(1, self.recurrent_chunk_size)):
			resps_list = self._unsqueeze_list(sequence_list)
			logits = super().forward(
				text_list=text_list,
				proms_list=proms_list,
				resps_list=resps_list,
				
				state=recurrent_state
			)

			r = super().sample(
				logits=logits,
				resps_list=resps_list,

				temperature=sampling_temperature,
				min_temperature=sampling_min_temperature,
				top_p=sampling_top_p,
				top_k=sampling_top_k,
				repetition_penalty=sampling_repetition_penalty,
				repetition_penalty_decay=sampling_repetition_penalty_decay,
				length_penalty=sampling_length_penalty,
				beam_width=sampling_beam_width,

				mirostat=mirostat,
			)

			if mirostat is not None:
				# r is the state
				mirostat = r
				# extract token from state
				r = [ state["token"] for state in mirostat ]
			# we do it here because the sampler will already expand our logits list
			elif sampling_beam_width > 0:
				# expand tuple
				r, s = r
				# first step, expand batch
				if batch_size == 1:
					batch_size *= sampling_beam_width
					text_list = text_list * sampling_beam_width
					proms_list = proms_list * sampling_beam_width
					sequence_list = sequence_list * sampling_beam_width
					stopped = torch.zeros(batch_size, device=device).bool()

				# update scores
				if sampling_beam_width_use_logs:
					scores = [ (math.log(scores[i]) if scores[i] > 0 else 0) + math.log(score) for i, score in enumerate(s) ]
				else:
					scores = [ scores[i] * score for i, score in enumerate(s) ]

			# append tokens
			for i, ri in enumerate(r):
				if self.stop_token in ri:
					stopped[i] = True
				sequence_list[i] = torch.cat([sequence_list[i], ri])

			# stop token found
			stopped |= r == self.stop_token
			if stopped.all().item():
				break

		# pick the best scoring candidate
		# desu this is always going to be candidate 0
		if sampling_beam_width and len(scores) > 0:
			best_idx, best_score = (0, 0)
			for idx, score in enumerate(scores):
				if best_score > score:
					best_idx, best_score = idx, score

			sequence_list = [sequence_list[best_idx]]

		return [self._prune(r) for r in sequence_list]


def example_usage():
	cfg.trainer.backend = "local"
	from functools import partial

	from einops import repeat

	from ..emb.qnt import decode_to_file, unload_model
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

	qnt = torch.load("data/qnt.pt")[0].t()[:, :cfg.models.prom_levels].to(device)

	text_list = [
		tokenize("ˈ a ɪ   w ɪ l   nˌ ɑː t  ˈ æ s k   ɐ   sˈ ɛ k ə n d   tˈ a ɪ m").to(device),
	]
	proms_list = [
		qnt[:75*3, :].to(device),
	]
	resps_list = [
		qnt.to(device),
	]

	text_list = text_list[:1]
	proms_list = proms_list[:1]
	resps_list = resps_list[:1]

	kwargs = {
		'n_tokens': 1024,
		'd_model': 1024, # 1536
		'n_heads': 16, # 24
		'n_layers': 12, # 32
	}
	
	"""
	try:
		kwargs['config'] = cfg.models.ar_nar
	except Exception as e:
		pass
	"""

	model = AR_NAR(**kwargs).to(device)
	#steps = 500
	#optimizer = ml.Prodigy(model.parameters(), lr=1.0)
	steps = 1000
	optimizer = ml.AdamW(model.parameters(), lr=1.0e-4)
	engine = Engine(model=model, optimizer=optimizer)

	torch.save( {
		'module': model.state_dict()
	}, "./data/test.pth" )

	print(f"AR+NAR parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
	
	@torch.inference_mode()
	def sample( name, steps=600 ):
		engine.eval()
		resps_list = engine(text_list, proms_list, max_steps=steps, sampling_temperature=0.95 )
		
		for i, o in enumerate(resps_list):
			_ = decode_to_file(o, f"data/ar.{i}.{name}.wav", device=device)

		resps_list = [r.unsqueeze(-1) for r in resps_list]
		resps_list = engine( text_list, proms_list, resps_list=resps_list, sampling_temperature=0.2 )

		for i, o in enumerate(resps_list):
			_ = decode_to_file(o, f"data/ar+nar.{i}.{name}.wav", device=device)

		unload_model()

	def train():
		engine.train()
		t = trange(steps)
		for i in t:
			stats = {"step": i}
			stats |= engine.traverse(text_list=text_list, proms_list=proms_list, resps_list=resps_list)

			tqdm.write(f"{stats}")

	sample("init", 5)
	train()
	sample("final")

if __name__ == "__main__":
	example_usage()
