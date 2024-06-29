"""
# an AR + NAR model that handles:
* inferencing the primary RVQ level in an autoregressive manner (AR)
* inferencing the remaining RVQ levels in parallel (NAR)

This model can fully handle being trained as a unified model (AR + NAR) or separate models (AR | NAR).
It's recommended to train as a unified model, then "distill" knowledge of each tasks separately, just in case.
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

from .lora import enable_lora

class AR_NAR(Base):
	@property
	def capabilities(self) -> list[str]:
		if hasattr(self, "config") and self.config:
			return self.config.capabilities
		return cfg.model.capabilities

	@property
	def quant_level_range(self) -> list[int]:
		if hasattr(self, "config") and self.config.rvq_level_range:
			return self.config.rvq_level_range
		return [ 0 if self.causal else 1, self.n_resp_levels ]

	@property
	def causal(self):
		return "ar" in self.capabilities

	@property
	def norm_type(self):
		return "ln" # if self.n_resp_levels == 1 else "adaln"

	@property
	def arch_type(self) -> str:
		if hasattr(self, "config") and self.config:
			return self.config.arch_type
		return cfg.model.arch_type

	@property
	def n_prom_levels(self) -> int:
		if hasattr(self, "config") and self.config:
			return self.config.prom_levels
		return cfg.model.prom_levels

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
	def p_rvq_levels(self) -> int:
		if hasattr(self, "config") and self.config:
			return self.config.p_rvq_levels
		return cfg.model.p_rvq_levels
	
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
	def interleave(self) -> bool:
		return False
	
	@property
	def monolithic(self) -> bool:
		return True

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

		# is training or NAR
		if resps_list is not None:
			n_levels_set = {r.shape[-1] for r in resps_list}
			n_levels = next(iter(n_levels_set))

			# is training
			if n_levels == self.n_resp_levels:
				# to-do: make this YAML configurable
				def sample_task():
					return "tts"

				# generate task list to train against
				task_list = [ sample_task() for _ in range(batch_size) ]

				# determines which RVQ level to target per batch
				quant_level_range = self.quant_level_range

				if self.p_rvq_levels == "equal":
					# randomly select a target RVQ-bin level (0 being AR, 1+ being NAR)
					quant_levels = [ random.randint(quant_level_range[0], quant_level_range[1] - 1) for i in range(batch_size) ]
				else: # if self.p_rvq_levels == "auto":
					# makes higher levels less likely
					def generate( lo=0, hi=8 ):
						index = lo
						p = random.random()
						for i in range(lo, hi):
							if p < 1.0 / (2 ** i):
								index = i
						return int(index)

					quant_levels = [ generate(quant_level_range[0], quant_level_range[1]) for i in range(batch_size) ]

				resps_list = [r[..., 0] if l == 0 else r[..., :l+1] for r, l in zip(resps_list, quant_levels)]
				
				# append stop tokens for AR
				# could technically do it in the .inputs call
				for i in range(batch_size):
					# only apply stop token for RVQ level 0
					if quant_levels[i] > 0:
						continue
					resps_list[i] = torch.cat([resps_list[i], torch.Tensor([self.stop_token]).to(device=device, dtype=torch.int16) ])

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
			
			# is NAR
			if max_levels == 0:
				max_levels = self.n_resp_levels - 1

			# expand if given a raw 1D tensor
			for i, resp in enumerate(resps_list):
				if resp.dim() == 1:
					resps_list[i] = resp.unsqueeze(-1)
			
			prev_list = resps_list

			for n in trange( max_levels, desc="NAR" ):				
				level = prev_list[0].shape[-1]
				if level >= max_levels + 1: # min(max_levels + 1, self.n_resp_levels): # commented out to experiment with exceeding trained levels
					break

				if cfg.lora is not None:
					enable_lora( self, cfg.lora.active_level( level ) )

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

				prev_list = [ torch.cat([rs, r.unsqueeze(-1).to(device)], dim=-1) for rs, r in zip(prev_list, resps_list) ]

			if cfg.lora is not None:
				enable_lora( self )

			return prev_list
		
		# is AR
		if cfg.lora is not None:
			enable_lora( self, cfg.lora.active_level( 0 ) )

		sequence_list = [ torch.zeros(0, device=device).to(torch.int16) for _ in range(batch_size) ]
		stopped = torch.zeros(batch_size, device=device).bool()
		
		stop_token = self.stop_token
		task_list = [ "tts" for _ in range(batch_size) ]

		state = None
		mirostat = [
			{"n": 1024, "tau": sampling_mirostat_tau, "eta": sampling_mirostat_eta, "max_surprise": sampling_mirostat_eta * 2, "error_surprise": 0, "running_total_surprise": 0}
		] * batch_size if sampling_mirostat_tau > 0.0 else None

		scores = [ 1.0 ] * sampling_beam_width

		if self.interleave:
			max_steps *= self.n_prom_levels

		# get next in sequence
		for n in trange(max_steps // max(1, self.causal_size), desc="AR"):
			# experimental rolling response to avoid too-long perplexity hits despite RetNet allegedly fixing this.
			# UNTESTED. In theory it would be better to also adjust the text, but there's no way of correlating text to segment of audio without something like wav2vec2
			if max_resp_context > 0:
				resps_list = self._unsqueeze_list([ sequence[-max_resp_context:] for sequence in sequence_list ] )
			else:
				resps_list = self._unsqueeze_list(sequence_list)

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

			if state is not None:
				logits, state = super().forward(
					inputs=inputs,
					state=state,
				)
			else:
				logits = super().forward(
					inputs=inputs,
					state=state,
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
					batch_size = sampling_beam_width
					text_list = text_list * sampling_beam_width
					proms_list = proms_list * sampling_beam_width
					sequence_list = sequence_list * sampling_beam_width
					stopped = torch.zeros(batch_size, device=device).bool()

				scores = [ scores[i] + score for i, score in enumerate(s) ]

			# append tokens
			for i, ri in enumerate(r):
				if stop_token in ri:
					stopped[i] = True
				sequence_list[i] = torch.cat([sequence_list[i], ri.to(device)])

			# stop token found
			stopped |= r == stop_token
			if stopped.all().item():
				break

		# pick the best scoring candidate
		# desu this is always going to be candidate 0
		if sampling_beam_width:
			sequence_list = [ sequence_list[0] ]

		sequence_list = [self._prune(r, stop_token) for r in sequence_list]
		return sequence_list


def example_usage():
	cfg.trainer.backend = "local"
	cfg.hyperparameters.gradient_accumulation_steps = 1
	if cfg.audio_backend == "dac":
		cfg.sample_rate = 44_000

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
		cfg.model.prom_levels = 1
		cfg.model.resp_levels = 1
	"""
	# cfg.model.loss_factors = {}

	def tokenize(content):
		return torch.tensor( cfg.tokenizer.encode(content) )

	def _load_quants(path) -> Tensor:
		qnt = np.load(path, allow_pickle=True)[()]
		return torch.from_numpy(qnt["codes"].astype(np.int16))[0, :cfg.model.prom_levels, :].t().to(torch.int16)

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

	model = AR_NAR(**kwargs).to(device)
	steps = 200 if cfg.model.arch_type in ["mamba","mamba2"] else 200

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

	print(f"AR+NAR ({cfg.model.arch_type}, {cfg.audio_backend}) parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

	@torch.inference_mode()
	def sample( name, steps=1000 ):
		if cfg.audio_backend == "dac" and name == "init":
			return

		engine.eval()
		if "ar" in cfg.model.capabilities:
			resps_list = engine(text_list, proms_list, max_steps=steps, sampling_temperature=0.95 )
		else:
			resps_list = [ qnt[:, 0].to( device ) ]

		if "nar" in cfg.model.capabilities:
			resps_list = engine( text_list, proms_list, resps_list=resps_list, sampling_temperature=0.2 )

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