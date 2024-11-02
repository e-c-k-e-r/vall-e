"""
# an AR model that (should) handle:
* handling all RVQ levels, but does it in an autoregressive manner

It's in a mess of a state, because I want this to be an interleaved model, but it just seems better to use the vall_e.models.experimental model.
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
import logging

_logger = logging.getLogger(__name__)

from ..emb.qnt import trim, encode_as_embedding

from .lora import enable_lora

def clamp(n, lo, hi):
	return max(lo, min(n, hi))

class AR(Base):
	def forward(
		self,
		text_list: list[Tensor],
		proms_list: list[Tensor],
		resps_list: list[Tensor] | None = None,
		
		task_list: list[Tensor] | None = None,
		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
		len_list: list[Tensor] | None = None,

		training: bool | int | None = None,

		max_steps: int = 1000,
		max_levels: int = 0,

		input_prompt_prefix: bool = False,
		prefix_silence: float = 1.0,

		sampling_temperature: float = 1.0,
		sampling_min_temperature: float = -1.0,
		sampling_top_k: int = -100,
		sampling_top_p: float = 1.0,
		sampling_min_p: float = 0.0,
		sampling_repetition_penalty: float = 1.0,
		sampling_repetition_penalty_decay: float = 0.0,
		sampling_length_penalty: float = 0.0,
		sampling_beam_width: int = 0,
		sampling_mirostat_tau: float = 0.0,
		sampling_mirostat_eta: float = 0.1,
		sampling_dry_multiplier=0.0,
		sampling_dry_base=1.75,
		sampling_dry_allowed_length=2,
		sampling_entropix=False,
		sampling_layer_skip: bool = False,
		sampling_layer_skip_exit_layer: int = -1,

		disable_tqdm=False,
		use_lora=None,
	):
		device = text_list[0].device
		batch_size = len(text_list)
		
		# generate task list if not provided
		if task_list is None:
			task_list = [ "tts" for _ in range(batch_size) ]

		# is training or NAR
		if resps_list is not None:
			n_levels_set = {r.shape[-1] for r in resps_list}
			n_levels = next(iter(n_levels_set))

			if training is None:
				training = n_levels == self.n_resp_levels

			# is training
			if training:
				# specifies how to sample probabilities of which RVQ levels to train against
				rvq_levels_p = self.config.experimental.rvq_levels_p if self.config is not None else "equal"
				# determines which RVQ level to target per batch
				quant_level_range = self.config.experimental.rvq_level_range if self.config is not None and self.config.experimental.rvq_level_range else [ 0 if self.causal else 1, self.n_resp_levels - 1 ]
				# rate to perform token dropout errors
				token_dropout_error = self.config.experimental.token_dropout_error
				# RVQ levels to apply token dropout on
				token_dropout_rvq_levels = self.config.experimental.token_dropout_rvq_levels
				# implicitly set it to all levels
				if not token_dropout_rvq_levels:
					token_dropout_rvq_levels = [0, self.resp_levels - 1]
				# allow passing a specific distribution of RVQ levels
				rvq_levels_p = rvq_levels_p if isinstance(rvq_levels_p, list) else []
				if not rvq_levels_p:
					lo, hi = quant_level_range[0], quant_level_range[1] + 1
					# randomly select a target RVQ-bin level (0 being AR, 1+ being NAR)
					if rvq_levels_p == "equal":
						rvq_levels_p = [ i for i in range( lo, hi ) ]
					else:
						# yuck
						rvq_levels_p = sum([[i for _ in range(hi - i)] for i in range( lo, hi ) ], [])

				# input RVQ levels
				if not self.interleave:
					quant_levels = [ random.choice( rvq_levels_p ) for i in range(batch_size) ]
					# trim resps to only contain all levels below the target level
					resps_list = [r[..., :l+1] for r, l in zip(resps_list, quant_levels)]
				else:
					quant_levels = [ 0 for i in range(batch_size) ]

				# tensor to cat for RVQ level 0
				# I hate python's value/reference semantics so much
				for i, quant_level, resps, proms in zip(range(batch_size), quant_levels, resps_list, proms_list):
					# cap quant_level if it exceeds its corresponding resp/prom
					if quant_level >= resps.shape[-1]:
						quant_levels[i] = resps.shape[-1] - 1

					# proms could be a Tensor, list[Tensor], or None
					if isinstance( proms, torch.Tensor ):
						if quant_level >= proms.shape[-1]:
							quant_levels[i] = proms.shape[-1] - 1

					elif isinstance( proms, list ):
						for j, prom in enumerate( proms ):
							if not isinstance( prom, torch.Tensor ):
								continue
						
						if quant_level >= prom.shape[-1]:
							quant_levels[i] = prom.shape[-1] - 1

					# apply token dropout error compensation
					if token_dropout_error > 0 and (token_dropout_rvq_levels[0] <= quant_level and quant_level <= token_dropout_rvq_levels[1]):
						steps = resps.shape[0]
						for l in range( quant_level ):
							for t in range( steps ):
								token = resps[t, l].item()

								if random.random() < token_dropout_error:								
									offset = 1 * ( 1 if random.random() < 0.5  else -1 )
									resps_list[i][t, l] = clamp(token + offset, 1, 1022) # +- 1

					# only apply stop token for RVQ level 0
					stop_sequence = torch.tensor([[self.stop_token] * resps.shape[-1]], device=device, dtype=torch.int16)
					resps_list[i] = torch.cat([ resps, stop_sequence ])
					

				inputs = self.inputs(
					text_list=text_list,
					proms_list=proms_list,
					resps_list=resps_list,
					lang_list=lang_list,
					tone_list=tone_list,
					task_list=task_list,
				)

				return super().forward(
					inputs=inputs,
				)
		
		# is AR
		if cfg.lora is not None:
			enable_lora( self, cfg.lora.active_level( 0 ) if use_lora is None else use_lora )

		sequence_list = [ torch.zeros(0, device=device).to(torch.int16) for _ in range(batch_size) ]
		stopped = torch.zeros(batch_size, device=device).bool()
		
		stop_token = self.stop_token


		state = None
		mirostat = [
			{"n": 1024, "tau": sampling_mirostat_tau, "eta": sampling_mirostat_eta, "max_surprise": sampling_mirostat_eta * 2, "error_surprise": 0, "running_total_surprise": 0}
		] * batch_size if sampling_mirostat_tau > 0.0 else None

		scores = [ 1.0 ] * sampling_beam_width

		# get next in sequence
		for n in trange(max_steps // max(1, self.causal_size), desc="AR", disable=disable_tqdm):
			resps_list = [x.unsqueeze(dim=-1) for x in sequence_list]

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

			output = super().forward(
				inputs=inputs,
				state=state,
			)
			logits, state = output.logits, output.state

			sampled = super().sample(
				logits=logits,
				prev_list=resps_list,

				temperature=sampling_temperature,
				min_temperature=sampling_min_temperature,
				top_p=sampling_top_p,
				top_k=sampling_top_k,
				min_p=sampling_min_p,
				repetition_penalty=sampling_repetition_penalty,
				repetition_penalty_decay=sampling_repetition_penalty_decay,
				length_penalty=sampling_length_penalty,
				beam_width=sampling_beam_width,

				mirostat=mirostat,

				dry_multiplier=sampling_dry_multiplier,
				dry_base=sampling_dry_base,
				dry_allowed_length=sampling_dry_allowed_length,
			)

			r = sampled[0]

			if mirostat is not None:
				mirostat = sampled.scores
			elif sampling_beam_width > 0:
				# expand tuple
				scores = sampled.scores
				# first step, expand batch
				if batch_size == 1:
					batch_size = sampling_beam_width
					text_list = text_list * sampling_beam_width
					proms_list = proms_list * sampling_beam_width
					sequence_list = sequence_list * sampling_beam_width
					stopped = torch.zeros(batch_size, device=device).bool()

				scores = [ scores[i] + score for i, score in enumerate(scores) ]

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

		for i, seq in enumerate( sequence_list ):
			steps = seq.shape[0] // self.n_resp_levels
			nearest_steps = steps * self.n_resp_levels
			sequence_list[i] = seq[:nearest_steps].view(( steps, self.n_resp_levels ))

		return sequence_list


def example_usage():
	cfg.trainer.backend = "local"
	cfg.hyperparameters.gradient_accumulation_steps = 1
	if cfg.audio_backend == "dac":
		cfg.sample_rate = 44_100

	from functools import partial
	from einops import repeat
	from tqdm import tqdm

	from ..emb.qnt import decode_to_file, unload_model, trim_random, repeat_extend_audio, concat_audio, merge_audio
	from ..engines import Engine, Engines
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
	noise = _load_quants(f"./data/noise.{'dac' if cfg.audio_backend == 'dac' else 'enc'}")

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

	batch_size = len(text_list)

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

	bos_id, space_id, eos_id = cfg.tokenizer.encode( " " )
	tasks = cfg.dataset.tasks_list

	model = AR(**kwargs).to(device)
	steps = 75 * len(tasks) * cfg.model.experimental.causal_size

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

	_logger.info(f"Optimizer: {optimizer}\tLearning rate: {learning_rate}")

	optimizer = optimizer(model.parameters(), lr=learning_rate)

	if scheduler == "schedulefree":
		if isinstance(optimizer, ml.AdamW):
			scheduler = ml.schedulefree.AdamWScheduleFree
		elif isinstance(optimizer, ml.SGD):
			scheduler = ml.schedulefree.SGDScheduleFree
		else:
			scheduler = None

		if scheduler is not None:
			_logger.info(f"Scheduler: {scheduler}")
			optimizer = scheduler( model.parameters(), lr = learning_rate )

	if cfg.optimizations.replace and cfg.optimizations.linear:
		model = ml.replace_linear( model )
		
	if cfg.optimizations.replace and cfg.optimizations.embedding:
		model = ml.replace_embedding( model )

	"""
	cfg.optimizations.model_offloading = {
		"devices": ["cuda:0", "cpu"],
	#	"limits": [ 0.9, -1 ],
		"assign": [[ f'layers.{i}.' for i in range(0,10) ], [ f'layers.{i}.' for i in range(11,12) ] + [ "model.norm" ]],
	#	"limits": [ 256 * (1024 ** 2), -1 ]
	}
	"""
	
	engine = Engine(model=model, optimizer=optimizer)
	engines = Engines({"ar": engine})
	engines.setup()
	
	"""
	if cfg.optimizations.model_offloading:
		model = ml.offload_model( model, policy=cfg.optimizations.model_offloading )
	"""

	"""
	torch.save( {
		'module': model.state_dict()
	}, f"./data/{cfg.model.arch_type}.pth" )
	"""

	_logger.info(f"AR ({cfg.model.arch_type}, {cfg.audio_backend}) parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

	@torch.no_grad()
	def sample_data(task=None):
		texts = []
		proms = []
		resps = []

		for i in range(batch_size):
			if task is None:
				task = random.choice(tasks)

			text = text_list[i]
			prom = proms_list[i]
			resp = resps_list[i]

			# do nothing
			if task == "tts":
				...
			elif task == "tts-c":
				trim_length = int(random.uniform(cfg.dataset.prompt_duration_range[0], cfg.dataset.prompt_duration_range[1]) * cfg.dataset.frames_per_second)

				prom = resp[:trim_length]
				resp = resp[trim_length:]
			elif task == "ns" or task == "sr":
				# extend the noise to fill the target audio
				noise_ext = repeat_extend_audio( noise, resp.shape[0] )
				# create the input prompt by merging the target audio with the noise
				prom = merge_audio( resp.cpu(), noise_ext, scale=[1, cfg.dataset.noise_scale], device=cfg.dataset.reencode_device )
				# set the target to just be the noise if <sr>
				if task == "sr":
					resp = noise_ext

				# set the text prompt to empty to train without a guided text prompt
				if random.random() < 0.5:
					text = torch.tensor([bos_id, eos_id], device=device, dtype=torch.uint8)

			texts.append( text.to(device) )
			proms.append( prom.to(device) )
			resps.append( resp.to(device) )

		return texts, proms, resps

	@torch.inference_mode()
	def sample( name, steps=1000, task=None ):
		engine.eval()

		texts, proms, resps = sample_data( task )

		resps = engine( texts, proms, max_steps=steps, sampling_temperature=0.95 )

		for i, o in enumerate(resps):
			_ = decode_to_file(o.to(dtype=torch.int32), f"data/{cfg.model.arch_type}.{cfg.audio_backend}.{i}.{task}.{name}.wav", device=device)

		unload_model()

	def train():
		engine.train()
		t = trange(steps)
		for i in t:
			texts, proms, resps = sample_data()

			stats = {"step": i}
			stats |= engine.traverse(text_list=texts, proms_list=proms, resps_list=resps)
			stats |= {"grad_norm": engine.get_global_grad_norm()}

			tqdm.write(f"{stats}")

		"""
		torch.save( {
			'module': model.state_dict()
		}, f"./data/{cfg.model.arch_type}.pth" )
		"""

	#sample("init", 5)
	train()

	"""
	if cfg.optimizations.compile:
		model = ml.compile_model(model, backend=cfg.optimizations.compile)
	"""
	
	for task in tasks:
		sample("final", task=task)

	engines.quit()

if __name__ == "__main__":
	example_usage()