from ..config import cfg

import torch
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from torch.nn import CrossEntropyLoss

import random
import math

from einops import rearrange
from tqdm import trange

AVAILABLE_ARCHES = []

try:
	from transformers import LlamaForCausalLM, LlamaConfig
	AVAILABLE_ARCHES.append("llama")
except Exception as e:
	print("Error importing `llama` arch:", e)
	pass

try:
	from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaConfig
	AVAILABLE_ARCHES.append("mamba")
except Exception as e:
	print("Error importing `mamba` arch:", e)
	pass

def _create_mask(l, device):
	seq = torch.arange(max(l), device=device).unsqueeze(0)  # (1 t)
	stop = torch.tensor(l, device=device).unsqueeze(1)  # (b 1)
	return (seq < stop).float()  # (b t)

def list_to_tensor(x_list: list[Tensor]):
	l = list(map(len, x_list))
	x = pad_sequence(x_list).t()

	m = _create_mask(l, x_list[0].device)
	m = m.to(x)
	return x, m

# fold into a typical LLM sequence (one embedding rather than split embeddings)
def fold(
	text_list = [],
	proms_list = [],
	resp_list = [],

	ignore_index = None,

	sep = 3,
	stop = 3,
	
	text_tokens = 256,
	audio_tokens = 1024,
	audio_rvq_levels = cfg.model.prom_levels
):

	device = text_list[0].device
	batch_size = len(text_list)
	input_ids = [ [] for _ in range(batch_size) ]

	offset = 0
	
	sep = torch.Tensor([ sep ])
	stop = torch.Tensor([ stop ])

	for i, text in enumerate(text_list):
		seq = text.to("cpu", dtype=torch.int64)
		input_ids[i].append( seq )
		input_ids[i].append( sep )
	
	offset = text_tokens
	for i, prom in enumerate(proms_list):
		if ignore_index is not None:
			seq = torch.Tensor( [ ignore_index for _ in range( prom.shape[0] * prom.shape[1] ) ] ).to("cpu", dtype=torch.int64)
		else:
			seq = prom.flatten().to("cpu", dtype=torch.int64)
			for idx, token in enumerate( seq ):
				token += offset + ( audio_tokens * ( idx % audio_rvq_levels ) )

		input_ids[i].append( seq )
		input_ids[i].append( sep )
	
	offset = text_tokens + (audio_tokens * audio_rvq_levels)
	for i, resp in enumerate(resp_list):
		seq = resp.flatten().to("cpu", dtype=torch.int64)
		for idx, token in enumerate( seq ):
			token += offset + ( audio_tokens * ( idx % audio_rvq_levels ) )
		input_ids[i].append( seq )
		input_ids[i].append( stop )

	for i, batch in enumerate(input_ids):
		input_ids[i] = torch.concat(input_ids[i], dim=-1).to(device=device, dtype=torch.int64)

	return list_to_tensor(input_ids)

# unfold from one unified token ID space to separate token spaces
def unfold(
	input_ids,

	sep = 3,
	stop = 3,
	
	text_tokens = 256,
	audio_tokens = 1024,
	audio_rvq_levels = cfg.model.prom_levels
):
	device = input_ids.device
	batch_size = input_ids.shape[0]

	text_list = [ [] for _ in range(batch_size) ]
	prom_list = [ [] for _ in range(batch_size) ]
	resp_list = [ [] for _ in range(batch_size) ]

	for i, batch in enumerate( input_ids ):
		for idx, token in enumerate( batch ):
			id = token.item()
			if id == sep or id == stop:
				continue

			if 0 <= id and id < text_tokens:
				text_list[i].append( id )
			elif text_tokens <= id and id < text_tokens + (audio_tokens * audio_rvq_levels):
				prom_list[i].append( (id - text_tokens) % audio_tokens )
			elif text_tokens + (audio_tokens * audio_rvq_levels) <= id:
				resp_list[i].append( (id - text_tokens) % audio_tokens )

		prom_len = len(prom_list[i])
		if prom_len % audio_rvq_levels == 0 and False:
			prom_list[i] = torch.Tensor(prom_list[i]).reshape( audio_rvq_levels, prom_len // audio_rvq_levels ).t()
		else:
			bins = [ [] for _ in range(audio_rvq_levels) ]
			for pos in range( prom_len ):
				rvq = pos % audio_rvq_levels
				bins[rvq].append( prom_list[i][pos] )
			nearest = ( len(bins) // audio_rvq_levels ) * audio_rvq_levels
			bins = bins[:nearest]
			prom_list[i] = torch.Tensor(bins).t().to(dtype=torch.int64)


		resp_len = len(resp_list[i])
		if len(resp_list[i]) % audio_rvq_levels == 0 and False:
			resp_list[i] = torch.Tensor(resp_list[i]).reshape( audio_rvq_levels, resp_len // audio_rvq_levels ).t()
		else:
			bins = [ [] for _ in range(audio_rvq_levels) ]
			for pos in range( resp_len ):
				rvq = pos % audio_rvq_levels
				bins[rvq].append( resp_list[i][pos] )
			nearest = ( len(bins) // audio_rvq_levels ) * audio_rvq_levels
			bins = bins[:nearest]
			resp_list[i] = torch.Tensor(bins).t().to(dtype=torch.int64)
		
		text_list[i] = torch.Tensor( text_list[i] ).to(dtype=torch.int64)

	return dict(
		text_list=text_list,
		prom_list=prom_list,
		resp_list=resp_list
	)


SELECTED_ARCH = cfg.model.arch_type 
if SELECTED_ARCH not in AVAILABLE_ARCHES:
	raise ValueError(f"Requesting arch `{SELECTED_ARCH}` but not available")

if SELECTED_ARCH == "mamba":
	LlmArchClass = MambaLMHeadModel
elif SELECTED_ARCH == "llama":
	LlmArchClass = LlamaForCausalLM
else:
	raise ValueError(f"Requesting arch `{SELECTED_ARCH}` but not available")

class Model(LlmArchClass):
	def __init__(
		self,
		d_model=1024,
		n_layers=12,
		n_heads=16,
		p_dropout=0.1,

		attention_backend=None,
		activation_checkpointing=True,
	):

		if SELECTED_ARCH == "llama":
			super().__init__(config=LlamaConfig(
				vocab_size=256 + (1024 * cfg.model.prom_levels) + (1024 * cfg.model.prom_levels) + 1,
				hidden_size=d_model,
				max_position_embeddings=cfg.dataset.frames_per_second * cfg.model.prom_levels * 60, # max-length of 60 seconds
				intermediate_size=d_model*4,
				num_hidden_layers=n_layers,
				num_attention_heads=n_heads,
				attention_dropout=p_dropout,
				num_key_value_heads=n_heads,
				sliding_window=cfg.dataset.frames_per_second * cfg.model.prom_levels * 12,
				hidden_act="gelu",
				is_encoder_decoder=False,
				is_decoder=True,
				attn_implementation=attention_backend,
			))
			
			if activation_checkpointing:
				self.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(
					use_reentrant=False
				))
		elif SELECTED_ARCH == "mamba":
			super().__init__(config=MambaConfig(
				vocab_size=256 + (1024 * cfg.model.prom_levels) + (1024 * cfg.model.prom_levels) + 1,
				d_model=d_model,
				n_layer=n_layers*2,
				#ssm_cfg={"layer": "Mamba2"},
			))


	def forward(
		self,
		*args,
		**kwargs,
	):
		output = super().forward(*args, **kwargs)

		if SELECTED_ARCH == "llama":
			if output.loss is not None:
				self.loss = dict(
					nll = output.loss,
				)
		elif SELECTED_ARCH == "mamba":
			if "labels" in kwargs:
				logits = output.logits
				labels = kwargs.pop("labels")

				# Shift so that tokens < n predict n
				shift_logits = logits[..., :-1, :].contiguous()
				shift_labels = labels[..., 1:].contiguous()
				# Flatten the tokens
				loss_fct = CrossEntropyLoss()
				shift_logits = shift_logits.view(-1, shift_logits.size(-1))
				shift_labels = shift_labels.view(-1)
				# Enable model parallelism
				shift_labels = shift_labels.to(shift_logits.device)
				loss = loss_fct(shift_logits, shift_labels)

				self.loss = dict(
					nll = loss,
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
		return torch.from_numpy(qnt["codes"].astype(np.int16))[0, :cfg.model.prom_levels, :].t().to(torch.int16)

	qnt = _load_quants(f"./data/qnt.{'dac' if cfg.audio_backend == 'dac' else 'enc'}")


	text_list = [
		tokenize("ˈaɪ wɪl nˌɑːt ˈæsk ɐ sˈɛkənd tˈaɪm").to(device),
		#tokenize("ˈaɪ wɪl nˌɑːt ˈæsk ɐ sˈɛkənd tˈaɪm").to(device),
	]
	proms_list = [
		qnt[:cfg.dataset.frames_per_second, :].to(device),
		#qnt[:cfg.dataset.frames_per_second, :].to(device),
	]
	resps_list = [
		qnt[:, :].to(device),
		#qnt[cfg.dataset.frames_per_second:, :].to(device),
	]

	text_list = text_list[:1]
	proms_list = proms_list[:1]
	resps_list = resps_list[:1]

	input_ids, attention_mask = fold(text_list, proms_list, resps_list)
	target_ids, target_attention_mask = fold(text_list, proms_list, resps_list, ignore_index=-100)
	prefix_input_ids, prefix_attention_mask = fold(text_list, proms_list)

	kwargs = {}
	model = Model(**kwargs).to(device)
	steps = 50

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
	}, f"./data/{SELECTED_ARCH}.pth" )

	print(f"{LlmArchClass} parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

	@torch.inference_mode()
	def sample( name, steps=cfg.model.prom_levels*cfg.dataset.frames_per_second*60 ):
		engine.eval()
		if SELECTED_ARCH == "mamba":
			output = model.generate(input_ids=prefix_input_ids, cg=True, max_length=steps, eos_token_id=3)
		else:
			output = model.generate(input_ids=prefix_input_ids, attention_mask=prefix_attention_mask, max_length=steps, eos_token_id=3, do_sample=False)

		unfolded = unfold( output )
		for i, batch in enumerate(unfolded["resp_list"]):
			_ = decode_to_file(batch.to(device=device), f"data/{SELECTED_ARCH}.{cfg.audio_backend}.{i}.{name}.wav", device=device)

		unload_model()

	def train():
		engine.train()
		t = trange(steps)
		for i in t:
			stats = {"step": i}
			if SELECTED_ARCH == "mamba":
				stats |= engine.traverse(input_ids=input_ids, labels=target_ids)
			else:
				stats |= engine.traverse(input_ids=input_ids, labels=target_ids, attention_mask=attention_mask)
			stats |= {"grad_norm": engine.get_global_grad_norm()}

			tqdm.write(f"{stats}")

		torch.save( {
			'module': model.state_dict()
		}, f"./data/{SELECTED_ARCH}.pth" )

	#sample("init", 5)
	train()
	sample("final")

if __name__ == "__main__":
	example_usage()
