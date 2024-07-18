"""
A (mostly) NAR model that handles inferencing all RVQ levels in parallel (NAR).
I believe Meta's Voicebox does this too (predict the utterance length, then decode in parallel)
It *does* have to inference the initial length in an autoregresssive-ish manner (it can technically also be done in parallel)

Initial experiments show this only really "works" for the a few brief seconds before going to silence. I imagine I need to read more papers or just need to train longer.
"""

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

class NAR(Base):
	@property
	def capabilities(self) -> list[str]:
		if hasattr(self, "config") and self.config:
			return self.config.capabilities
		return cfg.model.capabilities

	@property
	def causal(self):
		return "len" in self.capabilities

	@property
	def n_resp_levels(self) -> int:
		if hasattr(self, "config") and self.config:
			return self.config.resp_levels
		return cfg.model.resp_levels

	@property
	def n_max_levels(self) -> int:
		if hasattr(self, "config") and self.config:
			return self.config.max_levels
		return cfg.model.max_levels

	@property
	def n_tasks(self) -> int:
		if hasattr(self, "config") and self.config:
			return self.config.tasks
		return cfg.model.tasks

	@property
	def n_langs(self) -> int:
		if hasattr(self, "config") and self.config:
			return self.config.langs
		return cfg.model.langs

	@property
	def n_tones(self) -> int:
		if hasattr(self, "config") and self.config:
			return self.config.tones
		return cfg.model.tones

	@property
	def causal_size(self) -> int:
		# 1 for the stop token
		# governs how much to shift the logits by
		# could *technically* make it work to where it can also predict *ALL* RVQ levels in one step, but experimental.py is the better way to go about it
		return 1 # if self.causal else 0

	@property
	def version(self) -> int:
		if hasattr(self, "config") and self.config:
			return self.config.version
		return cfg.model.version

	def _prune(self, l: Tensor, stop = None):
		if stop is None:
			stop = self.stop_token
		indices = (l == stop).nonzero()
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
		
		task_list: list[Tensor] | None = None,
		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
		len_list: list[Tensor] | None = None,

		max_steps: int = 1000,
		max_levels: int = 0,
		max_resp_context: int = -1,

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

		# is training
		if resps_list is not None:
			p_len_task = 0.25

			n_levels_set = {r.shape[-1] for r in resps_list}
			n_levels = next(iter(n_levels_set))

			# assert n_levels == self.n_resp_levels

			# to-do: make this YAML configurable
			def sample_task():
				return "len" if random.random() < p_len_task else "tts"

			# generate task list to train against
			task_list = [ sample_task() for _ in range(batch_size) ]

			# determines which RVQ level to target per batch
			quant_level_range = self.config.experimental.rvq_level_range if self.config is not None and self.config.experimental.rvq_level_range else [ 0 if self.causal else 1, self.n_resp_levels ]

			p_rvq_levels = self.config.experimental.p_rvq_levels if self.config is not None else "equal"

			if p_rvq_levels == "equal":
				# randomly select a target RVQ-bin level (0 being AR, 1+ being NAR)
				quant_levels = [ random.randint(quant_level_range[0], quant_level_range[1] - 1) for i in range(batch_size) ]
			else: # if p_rvq_levels == "auto":
				# makes higher levels less likely
				def generate( lo=0, hi=8 ):
					index = lo
					p = random.random()
					for i in range(lo, hi):
						if p < 1.0 / (2 ** i):
							index = i
					return int(index)

				quant_levels = [ generate(quant_level_range[0], quant_level_range[1]) for i in range(batch_size) ]
			
			# clamp quant_levels because some of my audio was saved for only 8 out of 9 RVQ levels for DAC...
			for i in range(batch_size):
				# cap quant_level if it exceeds its corresponding resp/prom
				if quant_levels[i] >= resps_list[i].shape[-1]:
					quant_levels[i] = resps_list[i].shape[-1] - 1

				if quant_levels[i] >= proms_list[i].shape[-1]:
					quant_levels[i] = proms_list[i].shape[-1] - 1

			resps_list = [r[..., 0] if l == 0 else r[..., :l+1] for r, l in zip(resps_list, quant_levels)]

			inputs = self.inputs(
				text_list=text_list,
				proms_list=proms_list,
				resps_list=resps_list,
				lang_list=lang_list,
				tone_list=tone_list,
				task_list=task_list,

				quant_levels=quant_levels,
			)

			return super().forward(
				inputs=inputs,
				quant_levels=quant_levels,
			)

		# NAR
		if len_list is not None:
			# is NAR
			if max_levels == 0:
				max_levels = self.n_resp_levels
			
			# fill with mock tokens
			prev_list = [ torch.Tensor([ self.stop_token for _ in range(resp_len) ]).to(device=device, dtype=torch.int16) for resp_len in len_list ]

			start = True
			for n in trange( max_levels, desc="NAR" ):
				level = 0 if n == 0 else prev_list[0].shape[-1]
				if level >= max_levels + 1: # min(max_levels + 1, self.n_resp_levels): # commented out to experiment with exceeding trained levels
					break

				quant_levels = [ level for _ in range(batch_size) ] # torch.full((len(text_list),), level)

				inputs = self.inputs(
					text_list=text_list,
					proms_list=proms_list,
					resps_list=prev_list,
					lang_list=lang_list,
					tone_list=tone_list,
					quant_levels=quant_levels,
				)

				logits = super().forward(
					inputs=inputs,
					quant_levels=quant_levels,
				)

				"""
				resps_list = [ logit[-l:].argmax(dim=1) for logit, l in zip(logits, len_list) ]
				"""

				resps_list = super().sample(
					logits=logits,
					resps_list=prev_list,
					quant_levels=quant_levels,

					temperature=1.0 if n == 0 else sampling_temperature,
					min_temperature=sampling_min_temperature,
					top_p=sampling_top_p,
					top_k=sampling_top_k,
					repetition_penalty=sampling_repetition_penalty,
					repetition_penalty_decay=sampling_repetition_penalty_decay,
					#length_penalty=sampling_length_penalty,
					#beam_width=sampling_beam_width,
					#mirostat=mirostat,
				)

				if n == 0:
					prev_list = [ r.unsqueeze(-1).to(device) for r in resps_list ]
				else:
					prev_list = [ torch.cat([rs, r.unsqueeze(-1).to(device)], dim=-1) for rs, r in zip(prev_list, resps_list) ]

			return prev_list
		
		# is AR
		sequence_list = [ torch.Tensor([0]).to(device=device,dtype=torch.int16) for _ in range(batch_size) ]
		stopped = torch.zeros(batch_size, device=device).bool()
		
		stop_token = 10
		task_list = [ "len" for _ in range(batch_size) ]

		for n in trange(10, desc="AR"):
			len_list = sequence_list

			inputs = self.inputs(
				text_list=text_list,
				proms_list=proms_list,
				resps_list=resps_list,
				lang_list=lang_list,
				tone_list=tone_list,
				len_list=len_list,
				task_list=task_list,
				quant_levels=[ 0 for _ in range( max( batch_size, sampling_beam_width ) ) ]
			)

			logits = super().forward(
				inputs=inputs,
			)

			r = [ logit[-1:].argmax(dim=1) for logit in logits ]
			# sanitize
			for i, token in enumerate(r):
				if token > 10:
					r[i] = 0

			# append tokens
			for i, ri in enumerate(r):
				if stop_token in ri:
					stopped[i] = True
				sequence_list[i] = torch.cat([sequence_list[i], ri.to(device)])

			# stop token found
			stopped |= r == stop_token
			if stopped.all().item():
				break

		# convert tokens into int
		return [ int("".join([ str(token.item()) for token in r if token != stop_token ])) for r in sequence_list ]


def example_usage():
	cfg.trainer.backend = "local"
	cfg.hyperparameters.gradient_accumulation_steps = 1
	if cfg.audio_backend == "dac":
		cfg.sample_rate = 44_100

	from functools import partial
	from einops import repeat
	from tqdm import tqdm

	from ..emb.qnt import decode_to_file, unload_model
	from ..engines import Engine
	from ..utils import wrapper as ml
	
	import numpy as np
	import re

	device = "cuda"
	
	# mamba seems to ONLY be used as an AR (any NAR attempts lobotomizes it)
	"""
	if "mamba" in cfg.model.arch_type:
		cfg.model.resp_levels = 1
	"""
	# cfg.model.loss_factors = {}

	def tokenize(content):
		return torch.tensor( cfg.tokenizer.encode(content) )

	def _load_quants(path) -> Tensor:
		qnt = np.load(path, allow_pickle=True)[()]
		return torch.from_numpy(qnt["codes"].astype(np.int16))[0, :cfg.model.resp_levels, :].t().to(torch.int16)

	qnt = _load_quants(f"./data/qnt.{'dac' if cfg.audio_backend == 'dac' else 'enc'}")


	text_list = [
		tokenize("ˈaɪ wɪl nˌɑːt ˈæsk ɐ sˈɛkənd tˈaɪm").to(device),
		#tokenize("ˈaɪ wɪl nˌɑːt ˈæsk").to(device),
	]
	proms_list = [
		qnt[:cfg.dataset.frames_per_second, :].to(device),
		#qnt[:cfg.dataset.frames_per_second, :].to(device),
	]
	resps_list = [
		qnt[:, :].to(device),
		#qnt[:cfg.dataset.frames_per_second, :].to(device),
	]

	text_list = text_list[:1]
	proms_list = proms_list[:1]
	resps_list = resps_list[:1]

	# rentet-full is the only configuration with BitNet's BitLinear that converges despite the grad_norm saying otherwise
	kwargs = {
		'n_text_tokens': 256,
		'n_audio_tokens': 1024,

		'd_model': 1024, # 256, # 1024, # 1536
		'n_heads': 16, # 4, # 16, # 24
		'n_layers': 12, # 32
		'n_experts': 1,

		'p_dropout': 0.1,

		'l_padding': 8 if cfg.optimizations.fp8 else 0,

		'config': cfg.model
	}
	
	"""
	try:
		kwargs['config'] = cfg.model
	except Exception as e:
		pass
	"""

	model = NAR(**kwargs).to(device)
	steps = 500 

	optimizer = cfg.hyperparameters.optimizer.lower() if cfg.yaml_path is not None else "prodigy"
	scheduler = cfg.hyperparameters.scheduler.lower() if cfg.yaml_path is not None else ""
	learning_rate = cfg.hyperparameters.learning_rate if cfg.yaml_path is not None else None

	if cfg.optimizations.dadaptation:
		# do not combine the two
		if scheduler == "schedulefree":
			scheduler = ""

		learning_rate = 1.0
	
	if optimizer == "prodigy":
		if learning_rate is None:
			learning_rate = 1.0

		optimizer = ml.Prodigy
	elif optimizer == "adagrad":
		if learning_rate is None:
			learning_rate = 1.0e-2

		optimizer = ml.Adagrad
	elif optimizer == "adamw":
		if learning_rate is None:
			learning_rate = 1.0e-4

		optimizer = ml.AdamW
	elif optimizer == "sdg":
		if learning_rate is None:
			learning_rate = 1.0e-4

		optimizer = ml.SGD
	else:
		raise ValueError(f"Unrecognized optimizer: {optimizer}")

	print("Optimizer:", optimizer, "\tLearning rate:", learning_rate)

	optimizer = optimizer(model.parameters(), lr=learning_rate)

	if scheduler == "schedulefree":
		if isinstance(optimizer, ml.AdamW):
			scheduler = ml.schedulefree.AdamWScheduleFree
		elif isinstance(optimizer, ml.SGD):
			scheduler = ml.schedulefree.SGDScheduleFree
		else:
			scheduler = None

		if scheduler is not None:
			print("Scheduler:", scheduler)
			optimizer = scheduler( model.parameters(), lr = learning_rate )

	if cfg.optimizations.replace and cfg.optimizations.linear:
		model = ml.replace_linear( model )
		
	if cfg.optimizations.replace and cfg.optimizations.embedding:
		model = ml.replace_embedding( model )
	
	engine = Engine(model=model, optimizer=optimizer)

	"""
	torch.save( {
		'module': model.state_dict()
	}, f"./data/{cfg.model.arch_type}.pth" )
	"""

	print(f"NAR parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

	@torch.inference_mode()
	def sample( name, steps=1000 ):
		if cfg.audio_backend == "dac" and name == "init":
			return

		engine.eval()

		len_list = engine(text_list, proms_list, max_steps=steps, sampling_temperature=0.95 )
		resps_list = engine( text_list, proms_list, len_list=len_list, sampling_temperature=0.2 )

		for i, o in enumerate(resps_list):
			_ = decode_to_file(o.to(dtype=torch.int32), f"data/{cfg.model.arch_type}.{cfg.audio_backend}.{i}.{name}.wav", device=device)

		unload_model()

	def train():
		engine.train()
		t = trange(steps)
		for i in t:
			stats = {"step": i}
			stats |= engine.traverse(text_list=text_list, proms_list=proms_list, resps_list=resps_list)
			stats |= {"grad_norm": engine.get_global_grad_norm()}

			tqdm.write(f"{stats}")

		"""
		torch.save( {
			'module': model.state_dict()
		}, f"./data/{cfg.model.arch_type}.pth" )
		"""

	#sample("init", 5)
	train()
	sample("final")

if __name__ == "__main__":
	example_usage()