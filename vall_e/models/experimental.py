"""
This is an experiment to:
* entertain a thought to try and abide by HF's transformers API (to benefit from caching better)
* conform to a single embedding (instead of a bunch of them) by folding/unfolding inputs
* stop trying to make a mixed AR+NAR model work since it seems lobotomized if I keep trying to enforce both recurrent and parallel inferencing (despite a penalty cost)
	+ I will not cave and go with codebook patterns, not yet.
"""

from ..config import cfg

from ..data import fold_inputs, unfold_outputs

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy, MulticlassPrecision

import random
import math

from einops import rearrange
from tqdm import trange

from .arch import *

if cfg.model.arch_type not in AVAILABLE_ARCHES:
	raise ValueError(f"Requesting arch `{cfg.model.arch_type}` but not available")

if cfg.model.arch_type in ["mamba","mamba2"]:
	LlmArchClass = MambaLMHeadModel
elif cfg.model.arch_type == "llama":
	LlmArchClass = LlamaForCausalLM
elif cfg.model.arch_type == "retnet":
	LlmArchClass = RetNetForCausalLM
else:
	raise ValueError(f"Requesting arch `{cfg.model.arch_type}` but not available")

class Model(LlmArchClass):
	def __init__(
		self,

		n_text_tokens = 256,
		n_audio_tokens = 1024,

		d_model=1024,
		n_layers=12,
		n_heads=16,
		p_dropout=0.1,

		config = cfg.model,
	):
		self.hyper_config = config
		
		hf_attention = config.attention if config is not None else None
		gradient_checkpointing = config.gradient_checkpointing if config is not None else True
		# text_tokens + rvq levels + [audio tokens * codebooks] (prom) + [audio tokens * codebooks] (resp) + stop
		# vocab_size = n_text_tokens + cfg.model.max_levels + (n_audio_tokens * cfg.model.max_levels) + (n_audio_tokens * cfg.model.max_levels) + 1

		if hf_attention == "auto":
			if AVAILABLE_ATTENTIONS:
				hf_attention = AVAILABLE_ATTENTIONS[0]
			else:
				hf_attention = "eager"

		if hf_attention == "xformers":
			hf_attention = "mem_efficient"

		text_start = 0
		text_end = text_start + config.text_tokens

		lang_start = text_end
		lang_end = lang_start + config.langs

		rvq_start = lang_end
		rvq_end = rvq_start + config.resp_levels

		prom_start = rvq_end
		prom_end = prom_start + config.audio_tokens * config.resp_levels

		task_start = prom_end
		task_end = task_start + config.tasks

		tone_start = task_end
		tone_end = tone_start + config.tones
		
		resp_start = tone_end
		resp_end = resp_start + config.audio_tokens * config.resp_levels

		vocab_size = resp_end

		if config.arch_type == "llama":
			super().__init__(config=LlamaConfig(
				vocab_size=vocab_size,
				hidden_size=d_model,
				max_position_embeddings=cfg.dataset.frames_per_second * config.max_levels * 60, # max-length of 60 seconds
				intermediate_size=d_model*4,
				num_hidden_layers=n_layers,
				num_attention_heads=n_heads,
				attention_dropout=p_dropout,
				num_key_value_heads=n_heads,
				sliding_window=cfg.dataset.frames_per_second * config.max_levels * 12,
				hidden_act="gelu",
				is_encoder_decoder=False,
				is_decoder=True,
				attn_implementation=hf_attention,
			))
			
			if gradient_checkpointing:
				self.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(
					use_reentrant=False
				))
		elif config.arch_type == "retnet":
			super().__init__(config=RetNetConfig(
				vocab_size=vocab_size,
				decoder_embed_dim=d_model,
				decoder_value_embed_dim =d_model * 2,
				decoder_retention_heads=n_heads,
				decoder_ffn_embed_dim=d_model * 4,
				decoder_layers=n_layers,
				dropout=p_dropout,
				checkpoint_activations=gradient_checkpointing,
				activation_fn="gelu",
				use_layernorm=False,
				use_biases=False,
				use_glu=True,

				#chunkwise_recurrent=self.causal and self.recurrent_chunk_size > 0,
				#recurrent_chunkwise_size=self.recurrent_chunk_size if self.causal else 0,
				#no_output_layer=True,
				#rotary_embedding_base=self.rotary_embedding_base, # 10000

				decoder_normalize_before=True,
			))
		elif config.arch_type in ["mamba","mamba2"]:
			super().__init__(config=MambaConfig(
				vocab_size=vocab_size,
				d_model=d_model,
				n_layer=n_layers*2,
				d_intermediate=0, # d_model*4,
				ssm_cfg={"layer": "Mamba2", "use_mem_eff_path": True} if config.arch_type == "mamba2" else {},
				rms_norm=True,
				fused_add_norm=True,
				residual_in_fp32=False,
			))

			self.backbone.gradient_checkpointing = gradient_checkpointing

		self.accuracy_metric = None if True else MulticlassAccuracy(
			vocab_size,
			top_k=10,
			average="micro",
			multidim_average="global",
			ignore_index=-100,
		)

	def generate(
		self,
		*args,
		**kwargs
	):
		if cfg.model.arch_type in ["mamba","mamba2"]:
			kwargs["cg"] = True

			if "attention_mask" in kwargs:
				kwargs.pop("attention_mask")

			if "do_sample" in kwargs:
				kwargs.pop("do_sample")

			if "min_length" in kwargs:
				kwargs.pop("min_length")
			
			"""
			if "position_ids" in kwargs:
				kwargs.pop("position_ids")
			
			if "max_new_tokens" in kwargs:
				kwargs.pop("max_new_tokens")

			if "max_length" not in kwargs:
				kwargs["max_length"] = 500 * (self.hyper_config.resp_levels if self.hyper_config.experimental.interleave else 1)

			if "num_last_tokens" not in kwargs:
				kwargs["num_last_tokens"] = self.hyper_config.experimental.causal_size
			"""

		input_ids = kwargs.pop("input_ids")
		attention_mask = kwargs.pop("attention_mask", None)
		position_ids = kwargs.pop("position_ids", None)
		
		stop_token = kwargs.pop("eos_token_id", 3)
		max_steps = kwargs.pop("max_new_tokens", 500)
		
		device = input_ids.device
		batch_size = input_ids.shape[0]

		sequence_list = [ inputs for inputs in input_ids ]
		position_list = [ positions for positions in position_ids ]

		start_positions = [ inputs.shape[0] for inputs in input_ids ]

		stopped = torch.zeros(batch_size, device=device).bool()
		
		config = self.hyper_config
		state = None
		disable_tqdm = False
		causal_size = config.experimental.causal_size

		# get next in sequence
		for n in trange(max_steps // max(1, causal_size), desc="AR", disable=disable_tqdm):
			output = super().forward(
				input_ids=torch.stack(sequence_list),
				#attention_mask=attention_mask,
				#past_key_values=state,
				#position_ids=torch.stack(position_list),
				#use_cache=False,
				#return_dict=False
			)

			logits = output[0]
			# state = output[1]

			r = [ logit[-causal_size:].argmax(dim=1) for logit in logits ]

			# append tokens
			for i, ri in enumerate(r):
				if stop_token in ri:
					stopped[i] = True

				last_position_id = position_list[i][-1].item() + 1
				sequence_list[i] = torch.cat([ sequence_list[i], ri.to(device) ], dim=0)
				#position_list[i] = torch.cat([ position_list[i], torch.tensor([ last_position_id + _ for _ in range( ri.shape[0] ) ], device=device, dtype=torch.int32) ])

			# stop token found
			stopped |= r == stop_token
			if stopped.all().item():
				break

		def _prune(l: Tensor, stop = stop_token):
			indices = (l == stop).nonzero()

			if len(indices) == 0:
				return l

			return l[: indices.min().item()]

		sequence_list = [ _prune(seq[start_positions[i]:], stop_token) for i, seq in enumerate(sequence_list) ]
		return torch.stack(sequence_list)

		"""
		return super().generate(*args, **kwargs)
		"""

	def forward(
		self,
		*args,
		**kwargs,
	):
		config = self.hyper_config

		if "text_list" in kwargs:
			text_list = kwargs.pop("text_list", None)
			proms_list = kwargs.pop("proms_list", None)
			resps_list = kwargs.pop("resps_list", None)
			lang_list = kwargs.pop("lang_list", None)
			tone_list = kwargs.pop("tone_list", None)
			
			training = kwargs.pop("training", False)
			steps = kwargs.pop("steps", 500)
			
			batch_size = len(text_list)

			if training:
				quant_levels = None if config.experimental.interleave else [ random.randint( 0 if "ar" in config.capabilities else 1, config.max_levels - 1) for _ in range(batch_size) ]

				input_ids, attention_mask, position_ids = fold_inputs(
					text_list=text_list,
					prom_list=proms_list,
					resp_list=resps_list,
					targ_list=resps_list,
					quant_levels=quant_levels,
				)
				target_ids, target_attention_mask, target_position_ids = fold_inputs(
					text_list=text_list,
					prom_list=proms_list,
					resp_list=resps_list,
					targ_list=resps_list,
					quant_levels=quant_levels,
					ignore_index=-100
				)
				return self.forward(
					input_ids=input_ids,
					labels=target_ids,
					position_ids=position_ids,

					quant_levels=quant_levels,
				)
	
			if config.experimental.interleave:
				input_ids, attention_mask, position_ids = fold_inputs( text_list=text_list, prom_list=proms_list )
				output = self.generate(
					input_ids=input_ids,
					position_ids=position_ids,
					attention_mask=attention_mask,
					eos_token_id=3,
					do_sample=True,
					max_new_tokens=steps*config.max_levels,
				)
				return unfold_outputs( output )["resp_list"]

			resps_list = [ [] for _ in range(batch_size) ]
			for l in range(config.max_levels):
				quant_levels = [ l for _ in range(batch_size) ]

				input_ids, attention_mask, position_ids = fold_inputs(text_list=text_list, prom_list=proms_list, resp_list=resps_list, quant_levels=quant_levels)
				min_length = 1 
				for batch in input_ids:
					min_length = max( min_length, batch.shape[0] + 1 )

				# to-do: figure out a way to do one forward pass but sample N tokens to replicate the NAR sample pass
				output = self.generate(
					input_ids=input_ids,
					attention_mask=attention_mask,
					position_ids=position_ids,
					eos_token_id=3,
					do_sample=True,
					max_new_tokens=steps,
				)
				
				unfolded = unfold_outputs( output, quant_levels=quant_levels )

				if l == 0:
					steps = 0

				for batch, resp in enumerate(unfolded["resp_list"]):
					length = resp.shape[-1]

					# store length
					if l == 0:
						steps = max( steps, length )
					# pad
					else:
						resp = resp[:steps]
						if length < steps:
							resp = torch.cat([ resp, torch.Tensor([ 0 for _ in range(steps-length) ]).to(resp) ])
					resps_list[batch].append( resp )

			for i, resp in enumerate( resps_list ):
				resps_list[i] = torch.stack( resp ).t()

			return resps_list

		if config.arch_type in ["mamba","mamba2"]:
			kwargs.pop("attention_mask", None)

		labels = kwargs.pop("labels", None)
		quant_levels = kwargs.pop("quant_levels", None)

		output = super().forward(*args, **kwargs)
		logits = output.logits

		# i HATE the correct way
		if labels is not None:
			if quant_levels is None:
				quant_levels = [0 for _ in range(labels.shape[0])]

			# predict the next token for AR, else predict in place
			loss = sum([ F.cross_entropy(
				logit[:-config.experimental.causal_size, :] if quant_level == 0 or "nar" not in config.capabilities else logit,
				label[config.experimental.causal_size:] if quant_level == 0 or "nar" not in config.capabilities else label,
				ignore_index=-100
			) for logit, label, quant_level in zip( logits, labels, quant_levels ) ])

			self.loss = dict(
				nll = loss,
			)

			if self.accuracy_metric is not None:
				self.stats = dict(
					acc = (sum([ self.accuracy_metric( logit, target ) for logit, target in zip( logits, labels ) ] ) / len( logits )).item()
				)

			"""
			if config.loss_factors:
				sep = 3
				# determine specific sections to focus on
				indices = [ [ idx for idx, token in enumerate( batch ) if token == sep ] for i, batch in enumerate( labels ) ]

				text_index = 0
				resp_index = 1 # 1 includes everything non text, -3 includes pre_resp + resp (ignores prom, probably better to include prom here)

				labels_text = [ batch[:indices[i][text_index] + 1 ] for i, batch in enumerate( labels ) ]
				labels_resp = [ batch[indices[i][resp_index] + 1:] for i, batch in enumerate( labels ) ]

				logits_text = [ batch[:indices[i][text_index] + 1 ] for i, batch in enumerate( logits ) ]
				logits_resp = [ batch[indices[i][resp_index] + 1:] for i, batch in enumerate( logits ) ]

				loss_text = sum([ F.cross_entropy( logit[:-1, :], label[1:], ignore_index=-100 ) for logit, label in zip( logits_text, labels_text ) ]) / len(logits_text) * self.hyper_config.loss_factor("text")
				loss_resp = sum([ F.cross_entropy( logit[:-1, :], label[1:], ignore_index=-100 ) for logit, label in zip( logits_resp, labels_resp ) ]) / len(logits_resp) * self.hyper_config.loss_factor("resp")

				self.loss = dict(
					text = loss_text,
					resp = loss_resp,
				)

				if self.accuracy_metric is not None:
					self.stats = dict(
						acc = dict(
							text = (sum([ self.accuracy_metric( logit, target ) for logit, target in zip( logits_text, labels_text ) ] ) / len( logits_text )).item(),
							resp = (sum([ self.accuracy_metric( logit, target ) for logit, target in zip( logits_resp, labels_resp ) ] ) / len( logits_resp )).item(),
						)
					)
			"""

		return output

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

	def tokenize(content):
		return torch.tensor( cfg.tokenizer.encode(content) )

	def _load_quants(path) -> Tensor:
		qnt = np.load(path, allow_pickle=True)[()]
		return torch.from_numpy(qnt["codes"].astype(np.int16))[0, :cfg.model.max_levels, :].t().to(torch.int16)

	qnt = _load_quants(f"./data/qnt.{'dac' if cfg.audio_backend == 'dac' else 'enc'}")


	text_list = [
		tokenize("ˈaɪ wɪl nˌɑːt ˈæsk ɐ sˈɛkənd tˈaɪm").to(device),
		#tokenize("ˈaɪ wɪl nˌɑːt ˈæsk ɐ sˈɛkənd tˈaɪm").to(device),
	]
	prom_list = [
		qnt[:cfg.dataset.frames_per_second, :].to(device),
		#qnt[:cfg.dataset.frames_per_second, :].to(device),
	]
	resp_list = [
		qnt[:, :].to(device),
		#qnt[cfg.dataset.frames_per_second:, :].to(device),
		#qnt[:cfg.dataset.frames_per_second, :].to(device),
	]

	text_list = text_list[:1]
	prom_list = prom_list[:1]
	resp_list = resp_list[:1]

	kwargs = {}
	model = Model(**kwargs).to(device)
	steps = 50 # 100 if cfg.model.experimental.interleave else 300

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

	print(f"{LlmArchClass} parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

	@torch.inference_mode()
	def sample( name, steps=cfg.model.max_levels*cfg.dataset.frames_per_second*6 ):
		engine.eval()
		
		resp_list = model( text_list=text_list, proms_list=prom_list )

		for i, batch in enumerate(resp_list):
			_ = decode_to_file(batch.to(device=device), f"data/{cfg.model.arch_type}.{cfg.audio_backend}.{i}.{name}.wav", device=device)

		unload_model()

	def train():
		engine.train()
		t = trange(steps)
		for i in t:
			stats = {"step": i}

			stats |= engine.traverse(text_list=text_list, proms_list=prom_list, resps_list=resp_list, training=True)
			stats |= engine.gather_attribute("stats")
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
