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
import time
from einops import rearrange
from torch import Tensor
from tqdm import trange, tqdm

import logging

_logger = logging.getLogger(__name__)

from ..emb.qnt import trim, encode_as_embedding, get_silence
from ..utils import get_devices, setup_logging, timer, clamp, convert_kwargs

from .lora import enable_lora

text_task = [ "stt" ]

class AR_NAR(Base):
	def forward_train(
		self,
		text_list: list[Tensor],
		proms_list: list[Tensor],
		resps_list: list[Tensor],
		
		task_list: list[Tensor] | None = None,
		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
		len_list: list[Tensor] | None = None,
	):
		# deduce batch_size
		if text_list is not None:
			default_task = "tts"
			device = text_list[0].device
			batch_size = len(text_list)
		else:
			default_task = "stt"
			device = resps_list[0].device
			batch_size = len(resps_list)

		# specifies how to sample probabilities of which RVQ levels to train against
		rvq_levels_p = self.config.experimental.rvq_levels_p if self.config is not None else "equal"
		# determines which RVQ level to target per batch
		quant_level_range = self.config.experimental.rvq_level_range if self.config is not None and self.config.experimental.rvq_level_range else [ 0 if self.causal else 1, self.n_resp_levels - 1 ]
		# rate to perform token dropout errors
		token_dropout_error = self.config.experimental.token_dropout_error
		# RVQ levels to apply token dropout on
		token_dropout_rvq_levels = self.config.experimental.token_dropout_rvq_levels
		# RVQ levels to apply masking training on
		masking_train_rvq_levels = self.config.experimental.masking_train_rvq_levels

		# force set mask training
		if "len" not in self.capabilities:
			masking_train_rvq_levels = 0.0
		elif "ar" not in self.capabilities:
			masking_train_rvq_levels = 1.0

		# CFG
		cfg_text_dropout_p = self.config.experimental.cfg_text_dropout_p if self.config is not None else 0.0
		cfg_cond_dropout_p = self.config.experimental.cfg_cond_dropout_p if self.config is not None else 0.0
		cfg_prom_dropout_p = self.config.experimental.cfg_prom_dropout_p if self.config is not None else 0.0
		# rate to train RVQ level AR-ly or NAR-ly
		masking_train_p = self.config.experimental.masking_train_p if self.config is not None else 0.5
		# implicitly set it to all levels
		if not token_dropout_rvq_levels:
			token_dropout_rvq_levels = [0, self.resp_levels - 1]
		if not token_dropout_rvq_levels:
			token_dropout_rvq_levels = [0, 0]

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
		# timestep levels (for TTS NAR)
		timesteps = [ None for _ in range(batch_size) ]

		for i, task in enumerate( task_list ):
			lo, hi = masking_train_rvq_levels[0], masking_train_rvq_levels[1]
			if task in text_task:
				quant_levels[i] = 0 # self.n_resp_levels - 1
			elif lo <= quant_levels[i] and quant_levels[i] <= hi and random.random() < masking_train_p:
				timesteps[i] = random.random()
		
		# trim resps to only contain all levels below the target level
		resps_list = [r if t in text_task else r[..., :l+1] for r, l, t in zip(resps_list, quant_levels, task_list)]

		# tensor to cat for RVQ level 0
		text_stop_sequence = torch.tensor([2], device=device, dtype=torch.int16)
		text_start_stop_sequence = torch.tensor([1, 2], device=device, dtype=torch.int16)
		audio_stop_sequence = torch.tensor([[self.stop_token]], device=device, dtype=torch.int16)
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
					resps_list[i] = torch.cat([ resps, audio_stop_sequence ])

			if task == "len":
				quant_levels[i] = 0

			# apply CFG (should probably only apply to NAR quant level 0)
			if task not in text_task + ["len"]:
				drop_text = False
				drop_audio = False

				if random.random() < cfg_prom_dropout_p:
					drop_audio = True
				
				if random.random() < cfg_cond_dropout_p:
					drop_audio = True
					drop_text = True

				if drop_text:
					text_list[i] = text_start_stop_sequence

				if drop_audio:
					proms_list[i] = None

		inputs = self.inputs(
			text_list=text_list,
			proms_list=proms_list,
			resps_list=resps_list,
			lang_list=lang_list,
			tone_list=tone_list,
			task_list=task_list,
			time_list=timesteps,

			quant_levels=quant_levels,
		)

		return super().forward(
			inputs=inputs,
			quant_levels=quant_levels,
		)

	def forward_nar_masked(
		self,

		text_list: list[Tensor],
		proms_list: list[Tensor],
		resps_list: list[Tensor] | None = None,
		
		task_list: list[Tensor] | None = None,
		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
		len_list: list[Tensor] | None = None,

		disable_tqdm=False,
		use_lora=None,
		**sampling_kwargs,
	):
		device = text_list[0].device
		batch_size = len(text_list)

		# special "scheduling" to inference RVQ-level 0
		level = 0
		if cfg.lora is not None:
			enable_lora( self, cfg.lora.active_level( level ) if use_lora is None else use_lora )

		def log(x, eps = 1e-20):
			return torch.log(x.clamp(min = eps))

		def gumbel_sample(x, temperature = 1., dim = -1):
			return ((x / max(temperature, 1e-10)) + -log(-log(torch.zeros_like(x).uniform_(0, 1)))).argmax(dim = dim)

		# convert (N)AR specific args
		sampling_kwargs = convert_kwargs( sampling_kwargs, "ar_" )

		max_length = sampling_kwargs.pop("max_duration", 500)
		max_steps = sampling_kwargs.get("max_steps", 25)

		temperature = sampling_kwargs.pop("temperature", 1.0)
		cfg_strength = sampling_kwargs.get("cfg_strength", 0.0)
		start_noise = sampling_kwargs.get("denoise_start", 0.0)
		end_noise = sampling_kwargs.get("denoise_end", 1.0)
		max_steps = math.floor(max_steps * (end_noise - start_noise))

		len_list = [ clamp(l, 1, max_length) for l in len_list ]

		# if we're denoising from an existing sequence
		if start_noise > 0.0 and resps_list is not None:
			noise_p = math.cos( start_noise * math.pi * 0.5 )
			mask = [ torch.tensor( [ random.random() < noise_p for _ in range( seq_len ) ], dtype=torch.bool, device=device ) for seq_len in len_list ]
			resps_list = [ torch.where( mask, self.stop_token, resps[:, 0] ) for seq_len, resps in zip( len_list, resps_list ) ]
		else:
			resps_list = [ torch.ones((seq_len,), dtype=torch.int16, device=device) * self.stop_token for seq_len in len_list ]

		scores = [ torch.zeros((seq_len,), dtype=torch.float32, device=device) for seq_len in len_list ]
		quant_levels = [ level for _ in range(batch_size) ]
		null_text = [ torch.tensor([1, 2], device=device, dtype=torch.int16) for _ in range(batch_size) ]
		null_prom = [ None for _ in range(batch_size) ]
		prev_list = resps_list

		for timestep, steps_until_x0 in tqdm(zip(torch.linspace(start_noise, end_noise, max_steps), reversed(range(max_steps))), desc="NAR Masked", disable=disable_tqdm, total=max_steps):
			# get noise level, per cosine scheduling
			noise_p = math.cos( timestep * math.pi * 0.5 )
			# pick the worst scoring tokens to mask off
			masked_indices = [ score.topk( max(int( noise_p * seq_len ), 1), dim=-1 ).indices for score, seq_len in zip(scores, len_list) ]
			# mask off inputs
			resps_list = [ resp.scatter(0, indices, self.stop_token) for resp, indices in zip( resps_list, masked_indices ) ]
			# boolean mask
			is_masked = [ resps == self.stop_token for resps in resps_list ]

			time_list = [ timestep for _ in range(batch_size) ]

			# setup inputs
			inputs = super().inputs(
				text_list=text_list,
				proms_list=proms_list,
				resps_list=resps_list,
				lang_list=lang_list,
				tone_list=tone_list,
				time_list=time_list,
				quant_levels=quant_levels,
			)
			output = super().forward(
				inputs=inputs,
				quant_levels=quant_levels,
				#layer_skip_variables=sampling_layer_skip_variables,
			)

			logits = output.logits

			if cfg_strength > 0:
				null_inputs = super().inputs(
					text_list=null_text,
					proms_list=null_prom,
					resps_list=resps_list,
					lang_list=lang_list,
					tone_list=tone_list,
					time_list=time_list,
					quant_levels=quant_levels,
				)
				null_output = super().forward(
					inputs=null_inputs,
					quant_levels=quant_levels,
					#layer_skip_variables=sampling_layer_skip_variables,
				)
				for seq_len, logit, null_logit in zip(len_list, output.logits, null_output.logits):
					logit[-seq_len:] = null_logit[-seq_len:] + ( logit[-seq_len:] - null_logit[-seq_len:] ) * cfg_strength

			# sample with sampler settings
			filtered_sampled = super().sample(
				logits=logits,
				prev_list=prev_list,
				quant_levels=quant_levels,

				temperature=temperature * (steps_until_x0 / max_steps) ,
				**sampling_kwargs,
			)

			# retrieves unfiltered logits
			unfiltered_sampled = super().sample(
				logits=logits,
				prev_list=prev_list,
				quant_levels=quant_levels,

				temperature=0.0,
				**sampling_kwargs,
			)
			# update previous list of tokens
			prev_list = resps_list

			# sample with gumbelnoise
			# I actually feel like this doesn't matter? it's hard to judge with a partially trained NAR-len model
			sampled_ids = [ gumbel_sample( logits, temperature=temperature, dim=-1 ) for logits in filtered_sampled.logits[0] ]
			#sampled_ids = filtered_sampled[0]

			# keep unmasked tokens
			resps_list = [ torch.where( masked, input_ids, resps ) for masked, input_ids, resps in zip( is_masked, sampled_ids, resps_list ) ]
			# update scores (conjugated to put the worst scores at the top)
			scores = [ 1.0 - torch.tensor([score for score in scores], device=device) for scores in unfiltered_sampled.scores ]

		if cfg.experimental and max_steps > 0:
			print( timestep, steps_until_x0, noise_p, resps_list, scores )

		return resps_list

	def forward_nar(
		self,
		text_list: list[Tensor],
		proms_list: list[Tensor],
		resps_list: list[Tensor] | None = None,
		
		task_list: list[Tensor] | None = None,
		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
		len_list: list[Tensor] | None = None,

		disable_tqdm=False,
		use_lora=None,
		**sampling_kwargs,
	):
		# deduce batch_size
		if text_list is not None:
			default_task = "tts"
			device = text_list[0].device
			batch_size = len(text_list)
		else:
			default_task = "stt"
			device = resps_list[0].device
			batch_size = len(resps_list)


		max_levels = sampling_kwargs.get("max_levels", 0)
		# convert NAR specific args
		sampling_kwargs = convert_kwargs( sampling_kwargs, "nar_" )

		if max_levels == 0:
			max_levels = self.n_max_levels - 1

		"""
		sampling_layer_skip_variables = {} if sampling_layer_skip else None

		if sampling_layer_skip:
			if sampling_layer_skip_entropy_threshold >= 0:
				sampling_layer_skip_variables["entropy_threshold"] = sampling_layer_skip_entropy_threshold
			if sampling_layer_skip_varentropy_threshold >= 0:
				sampling_layer_skip_variables["varentropy_threshold"] = sampling_layer_skip_varentropy_threshold
			if sampling_layer_skip_exit_layer >= 0:
				sampling_layer_skip_variables["max_layer"] = sampling_layer_skip_exit_layer
		"""

		# inference NAR level 0
		if len_list is not None:
			resps_list = self.forward_nar_masked(
				text_list=text_list,
				proms_list=proms_list,
				resps_list=resps_list,
				task_list=task_list,
				lang_list=lang_list,
				tone_list=tone_list,
				len_list=len_list,
				**sampling_kwargs,				
			)

		# expand if given a raw 1D tensor
		for i, resp in enumerate(resps_list):
			if resp.dim() == 1:
				resps_list[i] = resp.unsqueeze(-1)
		
		prev_list = resps_list

		for n in trange( max_levels, desc="NAR", disable=disable_tqdm ):				
			level = prev_list[0].shape[-1]
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
				#layer_skip_variables=sampling_layer_skip_variables,
			)
			logits, state = output.logits, output.state

			sampled = super().sample(
				logits=logits,
				prev_list=prev_list,
				quant_levels=quant_levels,
				**sampling_kwargs,
			)

			resps_list = sampled[0]
			prev_list = [ torch.cat([rs, r.unsqueeze(-1).to(device=device)], dim=-1) for rs, r in zip(prev_list, resps_list) ]

		return prev_list

	def forward_ar(
		self,

		text_list: list[Tensor],
		proms_list: list[Tensor],
		resps_list: list[Tensor] | None = None,
		
		task_list: list[Tensor] | None = None,
		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
		len_list: list[Tensor] | None = None,
		disable_tqdm=False,
		use_lora=None,
		**sampling_kwargs,
	):
		# deduce batch_size
		if text_list is not None:
			default_task = "tts"
			device = text_list[0].device
			batch_size = len(text_list)
		else:
			default_task = "stt"
			device = resps_list[0].device
			batch_size = len(resps_list)

		if cfg.lora is not None:
			enable_lora( self, cfg.lora.active_level( 0 ) if use_lora is None else use_lora )

		# convert AR specific args
		sampling_kwargs = convert_kwargs( sampling_kwargs, "ar_" )

		temperature = sampling_kwargs.get("temperature", 1.0)
		min_temperature = sampling_kwargs.get("min_temperature", -1.0)
		max_duration = sampling_kwargs.get("max_duration", 500)
		beam_width = sampling_kwargs.get("beam_width", 0)
		entropix_sampling = sampling_kwargs.get("entropix_sampling", False)
		refine_on_stop = sampling_kwargs.get("refine_on_stop", False)
		input_prompt_prefix = sampling_kwargs.get("input_prompt_prefix", False)
		layer_skip = sampling_kwargs.get("layer_skip", False)
		prefix_silence = sampling_kwargs.get("prefix_silence", 0.0)
		mirostat_tau = sampling_kwargs.get("mirostat_tau", 0.0)
		mirostat_eta = sampling_kwargs.get("mirostat_eta", 0.0)

		# inference len
		if task_list is not None and task_list[0] == "len":
			sequence_list = [ torch.tensor([0], device=device,dtype=torch.int16) for _ in range(batch_size) ]
			stopped = torch.zeros(batch_size, device=device).bool()
			
			stop_token = 10
			task_list = [ "len" for _ in range(batch_size) ]
			quant_levels = [ 0 for _ in range( max( batch_size, beam_width ) ) ]

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
					quant_levels=quant_levels,
				)

				output = super().forward(
					inputs=inputs,
					quant_levels=quant_levels,
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

		# STT
		start_slice = [ 0 for _ in range(batch_size) ]
		sequence_list = [ torch.zeros(0, device=device).to(torch.int16) for _ in range(batch_size) ]
		stopped = torch.zeros(batch_size, device=device).bool()
		
		audio_stop_token = self.stop_token
		text_stop_token = 2

		state = None
		mirostat = [
			{"n": 1024, "tau": mirostat_tau, "eta": mirostat_eta, "max_surprise": mirostat_eta * 2, "error_surprise": 0, "running_total_surprise": 0}
		] * batch_size if mirostat_tau > 0.0 else None

		scores = [ 1.0 ] * beam_width
		metrics = []

		"""
		sampling_layer_skip_variables = {} if sampling_layer_skip else None

		if sampling_layer_skip:
			if sampling_layer_skip_entropy_threshold >= 0:
				sampling_layer_skip_variables["entropy_threshold"] = sampling_layer_skip_entropy_threshold
			if sampling_layer_skip_varentropy_threshold >= 0:
				sampling_layer_skip_variables["varentropy_threshold"] = sampling_layer_skip_varentropy_threshold
			if sampling_layer_skip_exit_layer >= 0:
				sampling_layer_skip_variables["max_layer"] = sampling_layer_skip_exit_layer
		"""

		for i, sequence in enumerate( sequence_list ):
			# add <bos> to text for STT
			if task_list[i] in text_task:
				start_slice[i] = 1
				sequence_list[i] = torch.cat([sequence_list[i], torch.tensor([1], dtype=torch.int16, device=device)])
			# treat input prompt as initial resp (by prefixing with the prompt instead)
			elif input_prompt_prefix:
				start_slice[i] = proms_list[i].shape[0]
				sequence_list[i], proms_list[i] = proms_list[i][:, 0], sequence_list[i]
			elif prefix_silence > 0:
				sequence_list[i] = get_silence(prefix_silence, device=sequence_list[i].device)
				sequence_list[i] = sequence_list[i][:, 0]
				# start_slice[i] = sequence_list[i].shape[0]

		# get next in sequence
		for n in trange(max_duration // max(1, self.causal_size), desc="AR", disable=disable_tqdm):
			# it would technically be faster to just append the new token's embedding to the inputs, but there's a VERY small performance gain from doing it, so it's not worth it
			text_list = [ sequence_list[i] if task in text_task else text_list[i] for i, task in enumerate(task_list) ]
			resps_list = [ sequence_list[i] if task not in text_task else resps_list[i] for i, task in enumerate(task_list) ]

			inputs = self.inputs(
				text_list=text_list,
				proms_list=proms_list,
				resps_list=resps_list,
				lang_list=lang_list,
				tone_list=tone_list,
				len_list=len_list,
				task_list=task_list,
				quant_levels=[ 0 for _ in range( max( batch_size, beam_width ) ) ]
			)

			# to-do: find an elegant way to write this
			output = super().forward(
				inputs=inputs,
				state=state,
				#layer_skip_variables=sampling_layer_skip_variables,
				output_attentions=entropix_sampling,
			)
			logits, state = output.logits, output.state

			sampled = super().sample(
				logits=logits,
				prev_list=[ resps_list[i] if task not in text_task else text_list[i] for i, task in enumerate( task_list ) ],
				**(sampling_kwargs | {"attentions": output.attentions if entropix_sampling else None}),
			)

			r = sampled[0]

			if cfg.experimental:
				if sampled.entropy:
					metrics.append( sampled.entropy )
				elif sampled.scores:
					#metrics.append( [ { "p": p[0], "exited_layer": output.exited_layer } for p in sampled.scores ] )
					metrics.append( [ { "p": p[0] } for p in sampled.scores ] )

			if mirostat is not None:
				mirostat = sampled.scores
			elif beam_width > 0:
				# expand tuple
				s = sampled.scores
				# first step, expand batch
				if batch_size == 1:
					batch_size = beam_width
					text_list = text_list * beam_width
					proms_list = proms_list * beam_width
					sequence_list = sequence_list * beam_width
					task_list = task_list * beam_width
					start_slice = start_slice * beam_width
					stopped = torch.zeros(batch_size, device=device).bool()

				scores = [ scores[i] + score for i, score in enumerate(s) ]

			# append tokens
			for i, ri in enumerate(r):
				task = task_list[i]
				stop_token = audio_stop_token if task not in text_task else text_stop_token
				if stop_token in ri:
					stopped[i] = True
				sequence_list[i] = torch.cat([sequence_list[i], ri.to(device)])

			# stop token found
			# stopped |= r == stop_token
			if stopped.all().item():
				break

		# to-do for layerskip / speculative sampling: rerun the last sequence again at max depth
		"""
		if metrics:
			from ..plot import plot_sample_metrics
			filename = "metrics"
			if entropix_sampling:
				filename += f'[entropix_sampling]'
			if sampling_layer_skip_exit_layer >= 0:
				filename += f'[{sampling_layer_skip_exit_layer+1}]'

			plot_sample_metrics( metrics, filename=f'{filename}.png' )
		"""

		# pick the best scoring candidate
		# desu this is always going to be candidate 0
		if beam_width:
			sequence_list = sequence_list[:1]
			task_list = task_list[:1]

		# remove stop token
		sequence_list = [self._prune(r, audio_stop_token if task_list[i] not in text_task else text_stop_token) for i, r in enumerate(sequence_list)]
		# remove <bos>
		sequence_list = [ sequence_list[i][start_slice[i]:] for i, task in enumerate( task_list ) ]

		if refine_on_stop:
			# get how much we need to slice from the end
			slice_lengths = [ sequence.shape[-1] for sequence in sequence_list ]
			# -1 for the stop token
			logits = [ logit[-length-1:-1] for logit, length in zip(logits, slice_lengths) ]
			# greedy sample from the sequence
			refined_list = [ logit.argmax(dim=-1) for logit in logits ]
			# to-do: compare scores
			# set the "refined" list as the output
			sequence_list = refined_list	

		return sequence_list

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

		disable_tqdm=False,
		use_lora=None,
		**sampling_kwargs,
	):
		# deduce batch_size
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

		# implicitly set for training
		if training is None and text_list is not None and resps_list is not None:
			n_levels_set = {r.shape[-1] for r in resps_list}
			n_levels = next(iter(n_levels_set))

			training = n_levels == self.n_resp_levels

		# is training
		if training:
			return self.forward_train(
				text_list=text_list,
				proms_list=proms_list,
				resps_list=resps_list,
				task_list=task_list,
				lang_list=lang_list,
				tone_list=tone_list,
				len_list=len_list,
			)

		# is NAR
		if (len_list is not None or resps_list is not None) and text_list is not None:
			return self.forward_nar(
				text_list=text_list,
				proms_list=proms_list,
				resps_list=resps_list,
				task_list=task_list,
				lang_list=lang_list,
				tone_list=tone_list,
				len_list=len_list,
				**sampling_kwargs,
			)

		# is AR
		return self.forward_ar(
			text_list=text_list,
			proms_list=proms_list,
			resps_list=resps_list,
			task_list=task_list,
			lang_list=lang_list,
			tone_list=tone_list,
			len_list=len_list,
			**sampling_kwargs,
		)


def example_usage():
	cfg.device = "cuda"
	cfg.trainer.backend = "local"
	if cfg.audio_backend == "dac":
		cfg.sample_rate = 44_100

	from functools import partial
	from einops import repeat
	from tqdm import tqdm

	from ..emb.qnt import decode_to_file, unload_model, trim_random, repeat_extend_audio, concat_audio, merge_audio
	from ..engines import Engine, Engines
	from ..utils import wrapper as ml
	from ..utils import setup_logging
	
	import numpy as np
	import re

	setup_logging()

	def load_artifact( path ):
		artifact = np.load(path, allow_pickle=True)[()]

		text = torch.tensor( cfg.tokenizer.encode( artifact["metadata"]["phonemes"] ) ).to(dtype=torch.uint8, device=cfg.device)
		audio = torch.from_numpy(artifact["codes"].astype(np.int16))[0, :, :].t().to(dtype=torch.int16, device=cfg.device)

		return text, audio

	text, audio = load_artifact(f"./data/qnt.{'dac' if cfg.audio_backend == 'dac' else 'enc'}")
	batch_size = cfg.hyperparameters.batch_size
	cfg.model.experimental.masking_train_p = 0.5

	text_list = [ text ] * batch_size
	proms_list = [ audio[:cfg.dataset.frames_per_second, :] ] * batch_size
	resps_list = [ audio ] * batch_size

	kwargs = {
		'n_text_tokens': 256,
		'n_audio_tokens': 1024,

		'd_model': 1024, # 256, # 1024, # 1536
		'n_heads': 16, # 4, # 16, # 24
		'n_layers': 12, # 32
		'n_experts': 1 if not cfg.model else cfg.model.experts,

		'p_dropout': 0.1,

		'l_padding': 8 if cfg.optimizations.fp8 else 0,

		'config': cfg.model
	}

	bos_id, space_id, eos_id = cfg.tokenizer.encode( " " )
	available_tasks = ["tts-ar", "tts-nar"]

	model = AR_NAR(**kwargs).to(cfg.device)
	steps = 500 // batch_size

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
	engines = Engines({"ar+nar": engine})
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

	_logger.info(f"AR+NAR ({cfg.model.arch_type}, {cfg.audio_backend}) parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

	@torch.no_grad()
	def sample_data(t=None):
		if isinstance(t, list):
			tasks = t
			texts = [ text_list[0].to(cfg.device) if task not in text_task else None for i, task in enumerate( tasks ) ]
			proms = [ proms_list[0].to(cfg.device) if task not in text_task else [ "stt" ] for i, task in enumerate( tasks ) ]
			resps = [ None if task not in text_task else resps_list[0].to(cfg.device) for i, task in enumerate( tasks ) ]

			return texts, proms, resps, tasks

		texts = []
		proms = []
		resps = []
		tasks = []

		for i in range(batch_size):
			task = random.choice(available_tasks) if t is None else t

			text = text_list[i].to(cfg.device)
			prom = proms_list[i].to(cfg.device)
			resp = resps_list[i].to(cfg.device)

			# do nothing
			if task == "stt":
				prom = [ task ]
			else:
				task = "tts" if random.random() > 0.1 else "len"

			texts.append( text )
			proms.append( prom )
			resps.append( resp )
			tasks.append( task )

		return texts, proms, resps, tasks

	@torch.inference_mode()
	def sample( name, steps=500, task=None ):
		engine.eval()

		text_list, proms_list, resp_list, task_list = sample_data( task )

		if task == "tts-nar":
			len_list = engine(text_list, proms_list, task_list=["len"], max_steps=5, temperature=0.0 )
			len_list = [ resp_list[0].shape[0] for l in len_list ]
			resps_list = engine( text_list, proms_list, len_list=len_list, temperature=0.0 )
		else:
			resps_list = engine( text_list, proms_list, task_list=["tts"], max_duration=steps, temperature=1.0 )
			resps_list = engine( text_list, proms_list, resps_list=resps_list, temperature=0.0 )

		for i, o in enumerate(resps_list):
			_ = decode_to_file(o.to(dtype=torch.int32), f"data/{cfg.model.arch_type}.{cfg.audio_backend}.{i}.{name}.{task}.wav", device=cfg.device)

		unload_model()

	def train():
		engine.train()
		t = trange(steps)
		for i in t:
			texts, proms, resps, tasks = sample_data()

			stats = {"step": i}
			stats |= engine.traverse(text_list=texts, proms_list=proms, resps_list=resps, task_list=tasks, training=True)
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
	
	for task in available_tasks:
		sample("final", task=task)

	engines.quit()

if __name__ == "__main__":
	example_usage()