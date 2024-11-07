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

from ..emb.qnt import trim, repeat_extend_audio

import logging

def clamp(n, lo, hi):
	return max(lo, min(n, hi))

_logger = logging.getLogger(__name__)

class NAR(Base):
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
		sampling_layer_skip_entropy_threshold: float = -1,
		sampling_layer_skip_varentropy_threshold: float = -1,

		sampling_refine_on_stop: bool = False,

		disable_tqdm=False,
		use_lora=None,
	):
		text_task = [ "stt" ]

		if text_list is not None:
			default_task = "tts"
			device = text_list[0].device
			batch_size = len(text_list)
		else:
			default_task = "stt"
			device = resps_list[0].device
			batch_size = len(resps_list)

		# generate task list if not provided
		if task_list is None:
			task_list = [ default_task for _ in range(batch_size) ]

		has_none = resps_list is None or text_list is None
		if not has_none:
			for i, task in enumerate( task_list ):
				if resps_list[i] is None or text_list[i] is None:
					has_none = True
					break

		# is training or NAR
		if not has_none:
			n_levels_set = {r.shape[-1] for r in resps_list}
			n_levels = next(iter(n_levels_set))

			# implicit
			if training is None:
				training = 0 if n_levels == self.n_resp_levels else None

			# is training
			if training is not None:
				len_train_p = self.config.experimental.len_train_p if self.config is not None else 0.05

				n_levels_set = {r.shape[-1] for r in resps_list}
				n_levels = next(iter(n_levels_set))

				# assert n_levels == self.n_resp_levels

				# to-do: make this YAML configurable
				def sample_task():
					return "len" if random.random() < len_train_p else "tts"

				# generate task list to train against
				task_list = [ sample_task() for _ in range(batch_size) ]

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
				quant_levels = [ random.choice( rvq_levels_p ) for i in range(batch_size) ]
				for i, task in enumerate( task_list ):
					if task in text_task:
						quant_levels[i] = 0 # self.n_resp_levels - 1
				
				# trim resps to only contain all levels below the target level
				resps_list = [r if t in text_task else r[..., :l+1] for r, l, t in zip(resps_list, quant_levels, task_list)]

				# tensor to cat for RVQ level 0
				text_stop_sequence = torch.tensor([[2] * 1], device=device, dtype=torch.int16)
				audio_stop_sequence = torch.tensor([[self.stop_token] * 1], device=device, dtype=torch.int16)
				# I hate python's value/reference semantics so much
				for i, quant_level, resps, proms, task in zip(range(batch_size), quant_levels, resps_list, proms_list, task_list):
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
					if quant_level <= 0:
						# append stop tokens for AR
						if task in text_task:
							#text_list[i] = torch.cat([ resps, text_stop_sequence ])
							...
						else:
							#resps_list[i] = torch.cat([ resps, audio_stop_sequence ])
							...

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


		if len_list is not None:
			# is NAR
			if max_levels == 0:
				max_levels = self.n_resp_levels
			
			# fill with mock tokens
			#prev_list = [ torch.tensor([ self.stop_token for _ in range(resp_len) ], device=device, dtype=torch.int16) for resp_len in len_list ]
			#prev_list = [ repeat_extend_audio( prom, resp_len ) for resp_len, prom in zip(len_list, proms_list) ]
			
			prev_list = [ torch.concat([ self.dropout_token.unsqueeze(0) for _ in range( resp_len ) ]) for resp_len in len_list ]
			#prev_list = [ None for resp_len in len_list ]

			# to-do: figure out why this fails when I copy some things from ar_nar
			for n in trange( max_levels, desc="NAR", disable=disable_tqdm ):
				level = 0 if n == 0 else prev_list[0].shape[-1]
				if level >= max_levels + 1: # min(max_levels + 1, self.n_resp_levels): # commented out to experiment with exceeding trained levels
					break

				if cfg.lora is not None:
					enable_lora( self, cfg.lora.active_level( level ) if use_lora is None else use_lora )

				quant_levels = [ level for _ in range(batch_size) ] # torch.full((len(text_list),), level)

				inputs = self.inputs(
					text_list=text_list,
					proms_list=proms_list,
					resps_list=prev_list,
					lang_list=lang_list,
					tone_list=tone_list,
					quant_levels=quant_levels,
				)

				output = super().forward(
					inputs=inputs,
					quant_levels=quant_levels,

				#	layer_skip_variables=sampling_layer_skip_variables,
				)
				logits, state = output.logits, output.state

				sampled = super().sample(
					logits=logits,
					prev_list=prev_list,
					quant_levels=quant_levels,

					#temperature=sampling_temperature,
					temperature=1.0 if n == 0 else sampling_temperature,
					min_temperature=sampling_min_temperature,
					top_p=sampling_top_p,
					top_k=sampling_top_k,
					min_p=sampling_min_p,
					repetition_penalty=sampling_repetition_penalty,
					repetition_penalty_decay=sampling_repetition_penalty_decay,
					#length_penalty=sampling_length_penalty,
					#beam_width=sampling_beam_width,
					#mirostat=mirostat,
				)
				resps_list = sampled[0]

				if n == 0:
					prev_list = [ r.unsqueeze(-1).to(device) for r in resps_list ]
				else:
					prev_list = [ torch.cat([rs, r.unsqueeze(-1).to(device)], dim=-1) for rs, r in zip(prev_list, resps_list) ]

			return prev_list
		
		# is AR
		if cfg.lora is not None:
			enable_lora( self, cfg.lora.active_level( 0 ) if use_lora is None else use_lora )

		sequence_list = [ torch.tensor([0], device=device,dtype=torch.int16) for _ in range(batch_size) ]
		stopped = torch.zeros(batch_size, device=device).bool()
		
		stop_token = 10
		task_list = [ "len" for _ in range(batch_size) ]

		for n in trange(10, desc="AR", disable=disable_tqdm):
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

			output = super().forward(
				inputs=inputs,
			)
			logits = output.logits

			r = [ logit[-1:].argmax(dim=1) for logit in logits ]
			# sanitize
			for i, token in enumerate(r):
				if token > 10:
					r[i][0] = stop_token

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
	steps = 250 

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
	
	engine = Engine(model=model, optimizer=optimizer)

	"""
	torch.save( {
		'module': model.state_dict()
	}, f"./data/{cfg.model.arch_type}.pth" )
	"""

	_logger.info(f"NAR parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

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