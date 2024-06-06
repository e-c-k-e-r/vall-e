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
		vocab_size = n_text_tokens + cfg.model.max_levels + (n_audio_tokens * cfg.model.max_levels) + (n_audio_tokens * cfg.model.max_levels) + 1

		if cfg.model.arch_type == "llama":
			super().__init__(config=LlamaConfig(
				vocab_size=vocab_size,
				hidden_size=d_model,
				max_position_embeddings=cfg.dataset.frames_per_second * cfg.model.max_levels * 60, # max-length of 60 seconds
				intermediate_size=d_model*4,
				num_hidden_layers=n_layers,
				num_attention_heads=n_heads,
				attention_dropout=p_dropout,
				num_key_value_heads=n_heads,
				sliding_window=cfg.dataset.frames_per_second * cfg.model.max_levels * 12,
				hidden_act="gelu",
				is_encoder_decoder=False,
				is_decoder=True,
				attn_implementation=hf_attention,
			))
			
			if gradient_checkpointing:
				self.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(
					use_reentrant=False
				))
		elif cfg.model.arch_type == "retnet":
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
		elif cfg.model.arch_type in ["mamba","mamba2"]:
			super().__init__(config=MambaConfig(
				vocab_size=vocab_size,
				d_model=d_model,
				n_layer=n_layers*2,
				ssm_cfg={"layer": "Mamba2", "chunk_size":64} if cfg.model.arch_type == "mamba2" else {},
				fused_add_norm=True,
				residual_in_fp32=True,
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

		return super().generate(*args, **kwargs)

	def forward(
		self,
		*args,
		**kwargs,
	):
		if cfg.model.arch_type in ["mamba","mamba2"]:
			if "attention_mask" in kwargs:
				kwargs.pop("attention_mask")

		labels = kwargs.pop("labels") if "labels" in kwargs else None

		output = super().forward(*args, **kwargs)
		logits = output.logits

		# i HATE the correct way
		if labels is not None:
			if self.hyper_config is None or not self.hyper_config.loss_factors:
				loss = sum([ F.cross_entropy( logit[:-1, :], label[1:], ignore_index=-100 ) for logit, label in zip( logits, labels ) ])
				self.loss = dict(
					nll = loss,
				)

				if self.accuracy_metric is not None:
					self.stats = dict(
						acc = (sum([ self.accuracy_metric( logit, target ) for logit, target in zip( logits, labels ) ] ) / len( logits )).item()
					)

			else:
				sep = 3
				# determine specific sections to focus on
				indices = [ [ idx for idx, token in enumerate( batch ) if token == sep ] for i, batch in enumerate( labels ) ]

				text_index = 0
				resp_index = 1 # 1 indluces everything non text, -3 includes pre_resp + resp (ignores prom, probably better to include prom here)

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

		return output

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
	steps = 100
	if cfg.model.arch_type == "mamba2":
		steps = 100
	elif cfg.model.arch_type == "llama":
		steps = 500
	elif cfg.model.interleave:
		steps = 250

	optimizer = cfg.hyperparameters.optimizer.lower() if cfg.cfg_path is not None else "prodigy"
	scheduler = cfg.hyperparameters.scheduler.lower() if cfg.cfg_path is not None else ""
	learning_rate = cfg.hyperparameters.learning_rate if cfg.cfg_path is not None else None

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

	torch.save( {
		'module': model.state_dict()
	}, f"./data/{cfg.model.arch_type}.pth" )

	print(f"{LlmArchClass} parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

	@torch.inference_mode()
	def sample( name, steps=cfg.model.max_levels*cfg.dataset.frames_per_second*6 ):
		engine.eval()
		batch_size = len(text_list)
		resp_list = None
		if cfg.model.interleave:
			input_ids, attention_mask = fold_inputs(text_list=text_list, prom_list=prom_list)
			output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=steps, eos_token_id=3, do_sample=False)
			
			unfolded = unfold_outputs( output )
			resp_list = unfolded["resp_list"]
		else:
			resp_list = [ [] for _ in range(batch_size) ]
			for l in range(cfg.model.max_levels):
				quant_levels = [ l for _ in range(batch_size) ]

				input_ids, attention_mask = fold_inputs(text_list=text_list, prom_list=prom_list, resp_list=resp_list, quant_levels=quant_levels)
				min_length = 1 
				for batch in input_ids:
					min_length = max( min_length, batch.shape[0] + 1 )

				output = model.generate(
					input_ids=input_ids,
					attention_mask=attention_mask,
					min_length=min_length,
					max_length=min_length+steps*2,
					eos_token_id=3,
					do_sample=False
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
					resp_list[batch].append( resp )

			for i, resp in enumerate( resp_list ):
				resp_list[i] = torch.stack( resp ).t()

		for i, batch in enumerate(resp_list):
			_ = decode_to_file(batch.to(device=device), f"data/{cfg.model.arch_type}.{cfg.audio_backend}.{i}.{name}.wav", device=device)

		unload_model()

	def train():
		engine.train()
		t = trange(steps)
		for i in t:
			stats = {"step": i}
			
			batch_size = len(text_list)
			quant_levels = None if cfg.model.interleave else torch.randint(0 if "ar" in cfg.model.capabilities else 1, cfg.model.max_levels, (batch_size,))
			if quant_levels is not None:
				resps_list = [ [] if l == 0 else resp for l, resp in zip(quant_levels, resp_list) ]
			else:
				resps_list = [ resp for resp in resp_list ]


			input_ids, attention_mask = fold_inputs(text_list=text_list, prom_list=prom_list, resp_list=resps_list, targ_list=resp_list, quant_levels=quant_levels)
			target_ids, target_attention_mask = fold_inputs(text_list=text_list, prom_list=prom_list, resp_list=resp_list, targ_list=resp_list, ignore_index=-100, quant_levels=quant_levels)
			
			stats |= engine.traverse(input_ids=input_ids, labels=target_ids, attention_mask=attention_mask)
			stats |= engine.gather_attribute("stats")
			stats |= {"grad_norm": engine.get_global_grad_norm()}

			tqdm.write(f"{stats}")

		torch.save( {
			'module': model.state_dict()
		}, f"./data/{cfg.model.arch_type}.pth" )

	#sample("init", 5)
	train()
	sample("final")

if __name__ == "__main__":
	example_usage()
