import copy
#import diskcache
import h5py
import json
import os
import subprocess
import sys
import time
import argparse
import yaml
import random
import logging
import itertools

import torch
import numpy as np

from dataclasses import asdict, dataclass, field

from functools import cached_property
from pathlib import Path

from .utils.distributed import world_size
from .utils.io import torch_load
from .utils import set_seed, prune_missing, md5_hash, coerce_dtype

@dataclass()
class BaseConfig:
	yaml_path: str | None = None # path passed in through --yaml

	@property
	def cfg_path(self):
		if self.yaml_path:
			return Path(self.yaml_path.parent)

		return Path(__file__).parent.parent / "data"

	@property
	def rel_path(self):
		return Path(self.cfg_path)

	@property
	def cache_dir(self):
		return self.rel_path / ".cache"

	@property
	def data_dir(self):
		return self.rel_path / "data"
	
	@property
	def metadata_dir(self):
		return self.rel_path / "metadata"

	@property
	def ckpt_dir(self):
		return self.rel_path / "ckpt"

	@property
	def log_dir(self):
		return self.rel_path / "logs" / str(self.start_time)

	@cached_property
	def start_time(self):
		return int(time.time())

	@cached_property
	def git_commit(self):
		try:
			cmd = "git rev-parse HEAD"
			return subprocess.check_output(cmd.split()).decode("utf8").strip()
		except:
			return ""

	@cached_property
	def git_status(self):
		try:
			cmd = "git status"
			return subprocess.check_output(cmd.split()).decode("utf8").strip()
		except:
			return ""

	def dumps(self):
		data = {k: getattr(self, k) for k in dir(self) if not k.startswith("__")}
		data = {k: v for k, v in data.items() if not callable(v)}
		return json.dumps(data, indent=2, default=str)

	def dump(self, path=None):
		if path is None:
			path = self.log_dir / "cfg.json"
		path.parent.mkdir(parents=True, exist_ok=True)
		with open(path, "w") as f:
			f.write(self.dumps())

	# ick
	@classmethod
	def prune_missing( cls, yaml ):
		default = cls(**{})
		default.format()
		yaml, missing = prune_missing( source=default, dest=yaml )
		if missing:
			_logger.warning(f'Missing keys in YAML: {missing}')
		return yaml

	@classmethod
	def from_yaml( cls, yaml_path ):
		state = {}
		state = yaml.safe_load(open(yaml_path, "r", encoding="utf-8"))
		state.setdefault("yaml_path", yaml_path)
		state = cls.prune_missing( state )
		return cls(**state)

	@classmethod
	def from_model( cls, model_path, lora_path=None ):
		if not model_path.exists():
			raise Exception(f'Model path does not exist: {model_path}')

		# load state dict and copy its stored model config
		model_kwargs = { "attention": "auto", "training": False, "teacher": False }
		model_state_dict = [ torch_load( model_path )["config"] | { "path": model_path } | model_kwargs ] if model_path and model_path.exists() else []
		lora_state_dict = [ torch_load( lora_path )["config"] | { "path": lora_path } ] if lora_path and lora_path.exists() else []

		state = { "models": model_state_dict, "loras": lora_state_dict, "trainer": { "load_state_dict": True } }
		return cls(**state)

	@classmethod
	def from_cli(cls, args=sys.argv):
		# legacy support for yaml=`` format
		for i, arg in enumerate(args):
			if arg.startswith("yaml"):
				args[i] = f'--{arg}'

		parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
		parser.add_argument("--yaml", type=Path, default=os.environ.get('VALLE_YAML', None)) # os environ so it can be specified in a HuggingFace Space too
		parser.add_argument("--model", type=Path, default=os.environ.get('VALLE_MODEL', None)) # os environ so it can be specified in a HuggingFace Space too
		parser.add_argument("--lora", type=Path, default=os.environ.get('VALLE_LORA', None)) # os environ so it can be specified in a HuggingFace Space too
		args, unknown = parser.parse_known_args(args=args)

		if args.model:
			return cls.from_model( args.model, args.lora )

		if args.yaml:
			return cls.from_yaml( args.yaml )

		return cls(**{})

	def __repr__(self):
		return str(self)

	def __str__(self):
		return self.dumps()

@dataclass()
class Dataset:
	training: list[Path] = field(default_factory=lambda: []) # paths to load into the training dataset
	validation: list[Path] = field(default_factory=lambda: []) # paths to load into the validation dataset
	noise: list[Path] = field(default_factory=lambda: []) # paths to load into the noise dataset
	
	# to-do: replace these since I feel this can be a bottleneck
	speaker_name_getter: str = "lambda p: f'{p.parts[-3]}_{p.parts[-2]}'" # function eval'd to extract a speaker's name from an utternace path
	speaker_group_getter: str = "lambda p: f'{p.parts[-3]}'" # function eval'd to extract a speaker's group from an utternace path
	# to-do: validate if I can ignore this since this is an artifact from when I only saved phonemes and encoded audio, and no metadata
	speaker_languages: dict = field(default_factory=lambda: {}) # dict where keys are the language codes and values are the speaker groups
	
	use_hdf5: bool = False # whether to load from an HDF5 dataset
	hdf5_name: str = "data.h5" # file name to load the HDF5 dataset
	hdf5_flag: str = "a" # flag to load the HDF5 file, automatically adjusted anyways
	
	use_metadata: bool = False # use genretaed metadata to aid in dataset loading
	
	validate: bool = True # validate each utterance on wheter it can be included based on duration range caps
	workers: int = 8 # number of dataloader workers to spawn
	cache: bool = True # use diskcache to cache the dataset

	min_utterances: int = 2 # minimum number of utterances a speaker can have
	max_utterances: int = 0 # max number of utterances a speaker can have (0 to disable)
	duration_range: list[float] = field(default_factory=lambda: [1.0, 12.0]) # the duration range an utterance can be to be included in the dataset
	
	sample_type: str = "path" # path | speaker
	sample_order: str = "interleaved" # duration
	sample_shuffle: bool = True # shuffles the indices in the sampler
	sample_max_duration_batch: float = 0.0 # total number of seconds of utterances per batched, 0 to disable
	# for a full sized model with 12GiB of VRAM for Encodec, 120 seconds is just enough
	# for a full sized model with 24GiB of VRAM for Encodec, 380 seconds is 80% VRAM consumed (but it might be limited by batch size)

	prompt_duration_range: list[float] = field(default_factory=lambda: [3.0, 6.0]) # the duration range the input prompts can be
	prompt_max_samples: int = 3 # maximum number of utterances that can be included in an input prompt for training
	prompt_continuous_utterance_p: float = 0.0 # probability to use the target utterance as an input prompt rather than using a different utterance
	prompt_similar_p: float = 0.75 # odds of sampling for a similar prompt instead of a random prompt
	prompt_similar_top_k: int = 1 # top-k similar candidates to sample from 
	prompt_similar_top_k_offset: int = 0 # offset from the top-k to sample from
	prompt_inject_noise_p: float = 0.0 # adds noise to the input prompt waveform to try and vary things
	
	resps_max_samples: int = 1 # number of samples to target for training
	resps_append_p: float = 1.0 # probability to append another sample to the training target
	resps_pad_silence_p: float = 0.0 # probability to pad resp with silence to fit within the next window

	tasks_list: list[str] = field(default_factory=lambda: ["tts"]) # list of tasks to train against
	reencode_on_concat: bool = False # whether to concat audio by decode => concat => encode, or naively concat codes
	reencode_device: str = "cpu" # "cpu" is slower but saves memory, cuda throws [rank0]: RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
	noise_scale: float = 0.25 # scaling noise value
	retokenize_text: bool = False

	_frames_per_second: int = 0 # allows setting your own hint

	def hash_key(self, *args):
		return md5_hash([ self.use_hdf5, self.min_duration, self.max_duration ] + [*args])

	@cached_property
	def frames_per_second(self):
		if self._frames_per_second > 0:
			return self._frames_per_second

		if cfg.audio_backend == "dac":
			if cfg.sample_rate == 44_100:
				return 87
			if cfg.sample_rate == 16_000:
				return 50
		
		if cfg.audio_backend == "nemo":
			return 86.1
		
		# 24Khz Encodec / Vocos and incidentally DAC are all at 75Hz
		return 75

	@property
	def min_phones(self):
		return self.phones_range[0]
	@property
	def max_phones(self):
		return self.phones_range[1]
	@property
	def min_duration(self):
		return self.duration_range[0]
	@property
	def max_duration(self):
		return self.duration_range[1]

# collection of experimental variables that should not be tampered with unless you know what you're doing
@dataclass()
class ModelExperimentalSettings:
	hf: bool = False # strictly utilizes a HF model and handles converting input IDs / outputs accordingly
	interleave: bool = False # use an interleaved AR rather than a split AR + NAR (worse performance and results due to everything being causal)
	split_classifiers: bool = False # each RVQ level gets its own classifier / output proj / LM head rather than sharing one for all RVQ levels (to-do: also split for text/prom)
	audio_embedding_sums: bool = False # whether each pass uses the previous RVQ codes or only the current level
	# a model trained not summing audio embeddings *can* have this enabled without any apparent issues
	# a model trained to sum *cannot* have this disabled without any apparent issues, or at least the ar+nar-retnet-8 can't.
	# in theory a model that is trained to sum embeddings can peform better due to "seeing" previous levles (due to the R in RVQ standing for residuals...), but in practice it seems fine to not do so
	
	audio_embedding_mode: str | None = None # None | "exclusive" | "inclusive", subjugates the audio backend's encoding/decoding model for embeddings, currently not used
	kv_heads: int = 0 # MHA or GQA (for supported backends)
	rvq_levels_p: str | list = "auto" # determines odds of selecting RVQ levels when training, "equal" will make each level equally likely
	rvq_level_range: list = field(default_factory=lambda: []) # some cringe to try and limit the RVQ training range for LoRAs, isn't necesary
	unified_position_ids: bool = True # False will generate position IDs partitioned for each section
	tie_classifier_to_embedding: bool = False # Ties the classifier output to their respective embeddings, this does not seem to do anything good in testing

	causal_size: int = 1 # experimental setting to see if I can just do parallel decoding in chunks instead of one-at-a-time without resorting to exotic solutions
	# VALL-E 2's approach of "combining token embeddings to group them" sounds terribad for a shared AR/NAR model
	# however, introducing partial parallel decoding for the AR maybe maybe MAYBE might help try and unify the AR/NAR tasks better, MAYBE
	# it just seems like a bitch to try and train something worthwhile with it, since there's crackles every other token
	# RetNet's chunked inferencing might be a better place for this

	masking_train_p: float = 0.0 # odds of training with masking
	masking_train_rvq_levels: list = field(default_factory=lambda: [0,0]) # determines which levels to do mask training on

	masking_ratio: str | float = 0.8 # sets a masking ratio, "random" will randomly pick, "rand" will pick between [0.2, 0.8]
	ignore_inputs_for_loss: bool = True # only calculate the loss on the outputs since thats what matters, as the inputs that do have loss calculated upon affects the loss for the entire sequence

	noncausal_masks: bool = False # to correct an oversight with Llama always using causal masks......
	classifiers_bias: bool = True # base LLaMAs do not bias the output heads, but my existing weights do
	max_position_embeddings: int = 70 * 65 * 5 # 5 minutes of audio

	# these technically should be as hyperparameters
	# performs token dropout to compensate for errors
	token_dropout_error: float = 0.0 # probability to nudge a token by ±1
	token_dropout_rate: float = 0.0 # probability to randomly set a token to a special dropout value
	token_dropout_rvq_levels: list = field(default_factory=lambda: [1,8]) # determines which levels to do dropout, by default do not do dropout on RVQ level 0
	# these technically should be as hyperparameters
	# classifier-free guidance training settings
	cfg_cond_dropout_p: float = 0.0 # 0.2 # probability to drop out text and audio during training
	cfg_text_dropout_p: float = 0.0 # 0.0  # probability to drop out input audio prompt during training
	cfg_prom_dropout_p: float = 0.0 # 0.3  # probability to drop out input audio prompt during training

	use_raw_text_p: float = 0.0 # probability to use raw text as the input prompt instead

	# failed experiment
	layerskip: bool = False # layerskip compatible model (or training for)
	#layerskip_rvq_levels: list = field(default_factory=lambda: []) # RVQ levels to train / inference layerskip for (to-do: implement, see if it matters)
	layerskip_r: int = 2 # number of layers to factor into early-exit loss calc
	layerskip_p_max: float = 0.1 # maximum probabilty to dropout the last layer, used for calculating layer dropout probabilities
	layerskip_e_scale: float = 0.2 # early-exit loss scalar value

# I really need to clean this up
@dataclass()
class Model:
	name: str = "ar+nar" # vanity name for the model
	version: int = 5 # 1 = old with MultiEmbedding, 2 = new with AudioEmbedding, 3+ = additional embeddings
	size: str | dict = "full" # preset string or explicitly defined dimensionality
	resp_levels: int = 8 # RVQ-bin levels this model supports
	tasks: int = 8 # ["tts", "ns", "sr", "tse", "cse", "nse"] and leaves two more for anything else I want (like "svc") (unused)
	langs: int = 1 # defined languages (semi-unused)
	tones: int = 1 # defined tones (unsued)
	experts: int = 1 # for mixtral / retnet-ts
	arch_type: str = "llama" # underling LM architecture used
	training: bool = False # I really need to attend to this
	teacher: bool = False # if this is to be treated as a teacher

	frozen_params: list[str] = field(default_factory=lambda: []) # frozen parameters that are not updated when training
	attention: str = "auto" # for llama arch_types: attention used
	dropout: float = 0.1 # adjustable dropout value
	path: Path | None = None
	#loss_factors: dict = field(default_factory=lambda: { "text": 0.1, "prom": 1.0, "resp": 1.0 }) # disable it by default since it causes a little more harm than good
	loss_factors: dict = field(default_factory=lambda: {})
	capabilities: list = field(default_factory=lambda: ["ar", "nar"]) # + ["lang", "tone"] if you have your dataset labeled for such
	kwargs: dict = field(default_factory=lambda: {})
	
	experimental: dict | ModelExperimentalSettings | None = None # experimental settings

	def get(self, name=None):
		return [ self ] if not name or self.name == name else []
	
	def loss_factor(self, k):
		return self.loss_factors.get(k, 0.0)

	@property
	def max_levels(self):
		# return RVQ level range
		if self.experimental is not None and self.experimental.rvq_level_range:
			return self.experimental.rvq_level_range[-1]
		return self.resp_levels

	@property
	# required for fp8 as the lengths needs to be divisible by 8
	def input_alignment(self):
		return 8 if cfg.optimizations.fp8 else 0

	@property
	def full_name(self):
		name = [ self.name ]
		
		if isinstance(self.size, dict):
			if hasattr(self.size, "label") and self.size['label']:
				name.append(f"{self.size['label']}")
		elif isinstance(self.size, str) and self.size not in ["full","extended"]:
			name.append(self.size)

		if self.experts > 1:
			name.append(f'{self.experts}x'+self.arch_type.replace("/", "-"))
		else:
			name.append(self.arch_type.replace("/", "-"))

		if cfg.optimizations.bitnet:
			name.append("bitnet")

		name.append(f'{self.resp_levels}')

		return "-".join(name)

	@property
	def tokens(self):
		return self.audio_tokens

	@property
	def audio_tokens(self):
		if isinstance(self.size, dict) and hasattr(self.size, "audio_tokens"):
			return self.size['audio_tokens']
		return 1024

	@property
	def text_tokens(self):
		if isinstance(self.size, dict) and hasattr(self.size, "text_tokens"):
			return self.size['text_tokens']
		return 256

	@property
	def dim(self):
		if isinstance(self.size, dict) and hasattr(self.size, "dim"):
			return self.size['dim']

		if isinstance(self.size, float):
			return math.floor(1024 * self.size)

		if self.size == "quarter":
			return 256
		if self.size == "half":
			return 512
		return 1024

	@property
	def heads(self):
		if isinstance(self.size, dict) and hasattr(self.size, "heads"):
			return self.size['heads']

		if isinstance(self.size, float):
			return math.floor(16 * self.size)

		if self.size == "quarter":
			return 4
		if self.size == "half":
			return 8
		return 16

	@property
	def layers(self):
		if isinstance(self.size, dict) and hasattr(self.size, "layers"):
			return self.size['layers']

		if self.size == "double":
			return 24
		if self.size == "extended":
			return 16
		return 12

	@property
	def activation_checkpointing(self):
		return cfg.trainer.activation_checkpointing
	
	@property
	def gradient_checkpointing(self):
		return cfg.trainer.gradient_checkpointing

	@property
	def lora_policy(self):
		include = ["model"] # by default only adapt the main model (not embeddings nor classifier/output projection/LM head/whatever)
		exclude = []

		if self.arch_type == "llama":
			include = ["self_attn", "mlp"] # target only the attention + mlp
			exclude = ["self_attn.k_proj"] # common literature says to ignore it
		if self.arch_type == "retnet":
			include = ["layers."] # target the core layers of the RetNet and ignore the auxiliary stuff
			exclude = ["retention.k_proj"] # attention-based transformers ignore the K, so might as well ignore it for the retnet

		return dict(include=include, exclude=exclude)

	# to-do: derive default arguments from here
	@property
	def get_kwargs(self, type):
		return self.kwargs

# should be renamed to Adapters
@dataclass()
class LoRA:
	name: str = "lora" # vanity name
	# to-do: find sane default values
	rank: int = 128 # rank for the LoRA
	alpha: int = 128 # rank for the LoRA
	training: bool = True # 
	embeddings: bool = False # train the embedding too
	parametrize: bool = False # whether to use the parameterized pathway for LoRAs or not
	rvq_levels: list[int] = field(default_factory=lambda: []) # determines RVQ levels to activate the LoRA
	path: Path | None = None

	@property
	def full_name(self):
		name = [ self.name, f"r{self.rank}", f"a{self.alpha}" ]
		return "-".join(name)

	# actually not needed anymore
	def active_level( self, level ):
		if not self.rvq_levels:
			return True
		return level in self.rvq_levels

@dataclass()
class Hyperparameters:
	batch_size: int = 8 # number of samples per training batch
	gradient_accumulation_steps: int = 32 # number of steps to accumulate gradients before updating
	gradient_clipping: int | float = 1.0 # largest size a gradient norm can be

	optimizer: str = "Adamw" # optimizer to use, should be 'Prodigyopt" now
	optimizer_params: dict = field(default_factory=lambda: {}) # to pass through deepspeed config
	
	learning_rate: float = 3.25e-4 # should be 1.0 for ProdigyOpt
	warmup_steps: int = 0 # number of steps to warm up the optimizer before performing updates, I think, this is just passed to deepspeed

	scheduler: str = "" # scheduler to use, currently don't ever use one so this doesn't really matter
	scheduler_type: str = "" # deprecated
	scheduler_params: dict = field(default_factory=lambda: {}) # to pass through deepspeed config

	autotune: bool = False # to do deepspeed's autotuning
	autotune_params: dict = field(default_factory=lambda: {}) # to pass through deepspeed config
	
	torch_optimizer: bool = False # if the requested optimizer is torch-derived rather than deepspeed supplied
	torch_scheduler: bool = False # if the requested scheduler is torch-derived rather than deepspeed-supplied

	teacher_alpha: float = 0.5 # mixing factor when performing knowledge distillation
	teacher_temperature: float = 1.0
	teacher_loss_fn: str = "mse" # kl | mse, use either kl_div or mse_loss (most implementations use kl, some literature says you can use mse)

@dataclass()
class Evaluation:
	batch_size: int = 64 # number of samples per batch during eval / val
	frequency: int = 250 # do eval / val every X iterations
	size: int = 64 # number of samples to generate during eval / val
	kwargs: dict = field(default_factory=lambda: {}) # inferencing kwargs

	# necessary in order to make it not confusing with requiring not-directyl exposed arguments passed to the model
	@cached_property
	def ar_kwargs( self ):
		return dict(
			max_steps=self.kwargs.get("max_ar_steps", 500),
			temperature=self.kwargs.get("ar_temperature", 1.0),
			min_temperature=self.kwargs.get("min_ar_temperature", -1),
			top_p=self.kwargs.get("top_p", 1.0), top_k=self.kwargs.get("top_k", 0), min_p=self.kwargs.get("min_p", 0.0),
			repetition_penalty=self.kwargs.get("repetition_penalty", 1.0), repetition_penalty_decay=self.kwargs.get("repetition_penalty_decay", 0),
			length_penalty=self.kwargs.get("length_penalty", 0),
			beam_width=self.kwargs.get("beam_width", 0),
			mirostat_tau=self.kwargs.get("mirostat_tau", 0),
			mirostat_eta=self.kwargs.get("mirostat_eta", 0),
			dry_multiplier=self.kwargs.get("dry_multiplier", 0),
			dry_base=self.kwargs.get("dry_base", 0),
			dry_allowed_length=self.kwargs.get("dry_allowed_length", 0),
			entropix=self.kwargs.get("entropix_sampling", False),
		)

	@cached_property
	def nar_kwargs( self ):
		return dict(
			max_levels=self.kwargs.get("max_nar_levels", 0),
			temperature=self.kwargs.get("nar_temperature", 0.0),
			min_temperature=self.kwargs.get("min_nar_temp", -1),
			top_p=self.kwargs.get("top_p", 1.0), top_k=self.kwargs.get("top_k", 0.0), min_p=self.kwargs.get("min_p", 0.0),
			repetition_penalty=self.kwargs.get("repetition_penalty", 1.0), repetition_penalty_decay=self.kwargs.get("repetition_penalty_decay", 0.0),
		)

@dataclass()
class DeepSpeed:
	zero_optimization_level: int = 0 # doesn't seem to work
	use_compression_training: bool = False # cope
	compression_bits: int = 8 # cope
	inferencing: bool = False # for using DeepSpeed's inferencing wrapper instead
	optimizer: bool = True # use DeepSpeed optimizer wrapper
	amp: bool = False # use DeepSpeed's AMP (requires some other package installed apparently)

	loss_scale_window: int = 100
	min_loss_scale: float = 8192.0

	config: dict = field(default_factory=lambda: {}) # to pass through deepspeed config

	@cached_property
	def ds_cfg(self):
		optimizer_params = cfg.hyperparameters.optimizer_params
		
		if 'lr' not in optimizer_params:
			optimizer_params["lr"] = cfg.hyperparameters.learning_rate,

		scheduler_params = cfg.hyperparameters.scheduler_params
		if 'warmup_num_steps' not in scheduler_params:
			scheduler_params['warmup_num_steps'] = cfg.hyperparameters.warmup_steps

		if 'total_num_steps' not in scheduler_params:
			scheduler_params['total_num_steps'] = cfg.trainer.iterations

		autotune_params = cfg.hyperparameters.autotune_params

		if "enabled" not in autotune_params:
			autotune_params['enabled'] = True
		
		if "results_dir" not in autotune_params:
			autotune_params['results_dir'] = str( cfg.rel_path / "autotune" / "results" )
		
		if "exps_dir" not in autotune_params:
			autotune_params['exps_dir'] = str( cfg.rel_path / "autotune" / "exps_" )

		# DeepSpeed fp16 is incompatible with its AMP
		if cfg.trainer.weight_dtype.lower() == "float16":
			self.amp = False

		# disable local AMP
		if self.amp:
			cfg.trainer.amp = False

		ds_cfg = {
			"train_micro_batch_size_per_gpu": cfg.hyperparameters.batch_size,
			"gradient_accumulation_steps": cfg.hyperparameters.gradient_accumulation_steps,
			"optimizer": {
				"type": cfg.hyperparameters.optimizer,
				"params": optimizer_params,
			} if not cfg.hyperparameters.torch_optimizer else None,
			"scheduler": {
				"type": cfg.hyperparameters.scheduler,
				"params": scheduler_params,
			} if not cfg.hyperparameters.torch_scheduler else None,
			"gradient_clipping": cfg.hyperparameters.gradient_clipping,
			"fp16": {
				"enabled": cfg.trainer.weight_dtype.lower() == "float16",
				"auto_cast": True, # ???
				"loss_scale_window": self.loss_scale_window, # raise every 100 consecutive good steps
				"min_loss_scale": self.min_loss_scale, # loss scale hitting 8K fries the model, 16K is fine but 32K is comfy
				"loss_scale": 0.0 if cfg.trainer.scale_loss else 1.0,
			},
			"bf16": {
				"enabled": cfg.trainer.weight_dtype.lower() == "bfloat16",
			},
			"amp": {
				"enabled": self.amp,
			},
			"autotuning": autotune_params if cfg.hyperparameters.autotune else None,
			"compression_training": {
				"weight_quantization": {
					"shared_parameters":{
						"enabled": True,
						"quantizer_kernel": True,
						"schedule_offset": 0,
						"quantize_groups": 64,
						"quantize_verbose": True,
						"quantization_type": "symmetric",
						"rounding": "nearest",
						"quantize_weight_in_forward": cfg.trainer.weight_dtype.lower() != "float16", #  MoQ (quantize in optimization step) weight quantization is only supported for FP16
						"fp16_mixed_quantize":{
							"enabled": False,
							"quantize_change_ratio": 1
						}
					},
					"different_groups": {
						"wq1": {
							"params": {
								"start_bits": self.compression_bits,
								"target_bits": self.compression_bits,
								"quantization_period": 0
							},
							"modules": [ "self_attn", "mlp" ] # for LLaMA, need to find for other arches
						}
					}
				},
				"activation_quantization": {
					"shared_parameters":{
						"enabled": True,
						"quantizer_kernel": True,
						"schedule_offset": 0,
						"quantize_groups": 64,
						"quantize_verbose": True,
						"quantization_type": "symmetric",
						"rounding": "nearest",
						"quantize_weight_in_forward": cfg.trainer.weight_dtype.lower() != "float16", #  MoQ (quantize in optimization step) weight quantization is only supported for FP16
						"fp16_mixed_quantize":{
							"enabled": False,
							"quantize_change_ratio": 1
						}
					},
					"different_groups": {
						"aq1": {
							"params": {
								"bits": self.compression_bits,
							},
							"modules": [ "self_attn", "mlp" ] # for LLaMA, need to find for other arches
						}
					}
				},
			} if self.use_compression_training else None,
			"zero_optimization": {
				"stage": self.zero_optimization_level,
				"contiguous_gradients": True,
				"overlap_comm": True,
				"reduce_scatter": True,
				"reduce_bucket_size": 5e8,
				"allgather_bucket_size": 5e8,
				"sub_group_size": 5e8,
				"round_robin_gradients": True,
				"offload_optimizer": {
					"device": "cpu",
					"pin_memory": True
				},
				"offload_param": {
					"device": "cpu",
					"pin_memory": True
				},
				"zero_quantized_weights": self.use_compression_training,
				"zero_hpz_partition_size": world_size(),
				"zero_quantized_gradients": self.use_compression_training,
			} if self.zero_optimization_level > 0 else None,
			"comms_logger": {
				"enabled": False
			}
		}

		null_keys = [ k for k in ds_cfg if not ds_cfg[k] ]
		for k in null_keys:
			del ds_cfg[k]

		if os.path.exists("./data/ds_config.json"):
			ds_cfg.update(json.loads(open("./data/ds_config.json", "r", encoding="utf-8")).read())
		else:
			ds_cfg.update(self.config)

		return ds_cfg

@dataclass()
class Trainer:
	iterations: int = 1_000_000 # maximum iterations to train

	save_tag: str = "step" # name to save checkpoints under, "step" will save as current step count
	load_tag: str | None = None # tag to load checkpoint from; if None: will check against contents of `./ckpt/{model-name}/latest` for the checkpoint name

	save_on_oom: bool = True # save if an OOM error is raised
	save_on_quit: bool = True # save when quitting training
	
	export_on_save: bool = False # export weights to local `fp32.pth` state_dict on saving a checkpoint
	export_on_quit: bool = False # export weights to local `fp32.pth` state_dict on quitting training
	
	save_frequency: int = 100 # frequency to save every X iterations

	keep_last_checkpoints: int = 0 # number of checkpoints to keep, prunes oldest ones

	load_state_dict: bool = False # loads `fp32.pth` state_dict, will automatically be done if a checkpoint is not found but `fp32.pth` exists
	load_states: bool = True #
	strict_loading: bool = False # sets strict_loading=True when loading the state dict
	load_module_only: bool = False # 
	restart_step_count: bool = False # clears the training stats when loading a checkpoint
	resize_modules: bool = True # automatically resizes 

	activation_checkpointing: bool | None = None # deprecated, should technically be used for only on activations and not the entire gradients, but HF only has gradient checkpointing
	gradient_checkpointing: bool = True # enables gradient checkpointing to save VRAM at the cost of slightly reduced performance when training

	check_for_oom: bool = True # checks for OOMs thrown during forward/backwards
	gc_mode: str | None = None # deprecated, but marks when to do GC

	wandb: bool = False # use wandb, if available

	weight_dtype: str = "float16" # dtype to have the model under

	amp: bool = False # automatic mixed precision
	ddp: bool = False # torch's internal DDP, automatically set if local backend is used and multiple GPUs are requested
	#scale_loss: bool = False # whether to perform loss scaling (for FP16 training) (it actually seems more harmful than not for this specific workload)

	load_webui: bool = False # load the web UI to allow inferencing during training, to-do: actually make this work

	backend: str = "local" # training backend to use. currently supports "local" | "deepspeed"
	deepspeed: DeepSpeed = field(default_factory=lambda: DeepSpeed) # deepspeed settings

	@cached_property
	def dtype(self):
		return coerce_dtype(self.weight_dtype)

	@cached_property
	def scale_loss(self):
		# currently cannot feasibly apply loss scaling with DeepSpeed backend (it can handle it itself anyways)
		return self.dtype == torch.float16

@dataclass()
class Inference:
	backend: str = "local" # backend to use when inferencing
	weight_dtype: str = "float16" # dtype to load the model under
	amp: bool = True # automatic mixed precision during inferencing

	normalize: bool = False # to-do: actually normalize input / output audio, I believe this might cause issues though

	batch_size: int = 16 # I don't know what would be a good batch size

	@property
	def dtype(self):
		return coerce_dtype(self.weight_dtype)

@dataclass()
class Optimizations:
	injects: bool = False # overwrites default torch classes (not recommended)
	replace: bool = False # replaces modules in place with the optimized version (recommended)
	compile: bool | str = False # runs torch.compile on the model

	linear: bool = True # inject/replace linear for BnB
	embedding: bool = True # inject/replace embedding for BnB
	optimizers: bool = True # inject/replace optimizers (BnB, DAdaptation)
	
	bitsandbytes: bool = False # use bitsandbytes
	dadaptation: bool = False # use dadaptation optimizer
	bitnet: bool = False # use bitnet
	fp8: bool = False # use fp8

	# to-do: validate this madness works still, I don't remember what schizodemon told me to do this
	model_offloading: dict | None = None # automatically splits the model over a list of devices
	# example: {"include":["model"], "limits": [ (6 * 1024) * (1024 ** 2), -1 ]} will have the GPU capped to 6GiB, and offload the remaining layers to CPU
	# example: {"include":["model"], "device": ["cuda:0", "cuda:1"], "limits": [ 0.5, 0.5 ]} will have the GPU 1 try and use 50% of the model, and GPU 2 try and use the other 50%
	# | {"assign": [[ f'layers.{i}.' for i in range(0,6) ], [ f'layers.{i}.' for i in range(6,12) ]]} will assign layers 0-5 to device 1, and 6-12 to device 2

	tensorrt: bool = False
	unsloth: bool = False # unsloth gradient checkpointing (it just offloads tensors to the CPU during backwards, I don't think it's significant enough to bother with on small models)

@dataclass()
class Config(BaseConfig):
	device: str = "cuda" # target device
	mode: str = "training" # "inferencing"
	experimental: bool = False # debug flag
	silent_errors: bool = False # if False, raise exceptions on errors that could silently lead to problems, if True ignore them

	dataset: Dataset = field(default_factory=lambda: Dataset)
	models: dict | list | None = field(default_factory=lambda: [])
	loras: dict | list | None = field(default_factory=lambda: [])
	hyperparameters: Hyperparameters = field(default_factory=lambda: Hyperparameters)
	evaluation: Evaluation = field(default_factory=lambda: Evaluation)
	trainer: Trainer = field(default_factory=lambda: Trainer)
	inference: Inference = field(default_factory=lambda: Inference)
	optimizations: Optimizations = field(default_factory=lambda: Optimizations)
	
	tokenizer: str | None = None # tokenizer class
	tokenizer_path: str = "./tokenizer.json" # tokenizer path
	
	text_tokenizer: str | None = None # tokenizer class
	text_tokenizer_path: str = "./text_tokenizer.json" # tokenizer path

	sample_rate: int = 24_000 # sample rate the model expects
	audio_backend: str = "vocos" # audio backend to use "encodec" | "vocos" | "dac""

	weights_name: str = "fp32"
	weights_format: str = "sft" # "pth" | "sft"
	supported_weights_formats: list[str] = field(default_factory=lambda: ["sft", "safetensors", "pt", "pth"])

	def set_audio_backend(self, audio_backend):
		cfg.audio_backend = audio_backend
		audio_extension = None
		if audio_backend in ["encodec", "vocos"]:
			audio_extension = ".enc"
			cfg.sample_rate = 24_000
			cfg.model.resp_levels = 8
		elif audio_backend == "dac":
			audio_extension = ".dac"
			cfg.sample_rate = 44_100
			cfg.model.resp_levels = 9
		elif cfg.audio_backend == "audiodec":
			audio_extension = ".dec"
			cfg.sample_rate = 48_000
			cfg.model.resp_levels = 8 # ?
		elif cfg.audio_backend == "nemo":
			audio_extension = ".nem"
			cfg.sample_rate = 44_100
			cfg.model.resp_levels = 8
			#cfg.model.audio_tokens = 1000
		else:
			raise Exception(f"Unknown audio backend: {audio_backend}")

	@property
	def audio_backend_extension(self):
		audio_extension = None
		if self.audio_backend in ["encodec", "vocos"]:
			audio_extension = ".enc"
		elif self.audio_backend == "dac":
			audio_extension = ".dac"
		elif self.audio_backend == "audiodec":
			audio_extension = ".dec"
		elif self.audio_backend == "nemo":
			audio_extension = ".nem"
		return audio_extension

	@property
	def model(self):
		for i, model in enumerate(self.models):
			if model.training:
				return model

		return self.models[0] if len(self.models) > 0 else None

	# should be renamed to adapters
	@property
	def lora(self):
		for i, lora in enumerate(self.loras):
			if lora.training:
				return lora

		return self.loras[0] if len(self.loras) > 0 else None

	@property
	def distributed(self):
		return world_size() > 1

	@cached_property
	def get_spkr(self):
		return eval(self.dataset.speaker_name_getter)

	@cached_property
	def get_spkr_group(self):
		return eval(self.dataset.speaker_group_getter)

	"""
	@cached_property
	def diskcache(self):
		if self.yaml_path is not None and self.dataset.cache:
			return diskcache.Cache(self.cache_dir).memoize
		return lambda: lambda x: x
	"""

	# this gets called from vall_e.inference
	def load_yaml( self, config_path ):
		tmp = Config.from_yaml( config_path )
		self.__dict__.update(tmp.__dict__)
	
	def load_model( self, config_path, lora_path=None ):
		tmp = Config.from_model( config_path, lora_path )
		self.__dict__.update(tmp.__dict__)

	def load_hdf5( self, write=False ):
		if hasattr(self, 'hdf5'):
			self.hdf5.close()

		if self.distributed:
			self.dataset.hdf5_flag = "r"
		try:
			self.hdf5 = h5py.File(f'{self.rel_path}/{self.dataset.hdf5_name}', 'a' if write else self.dataset.hdf5_flag) # to-do, have an easy to set flag that determines if training or creating the dataset
		except Exception as e:
			_logger.warning(f"Error while opening HDF5 file: {self.rel_path}/{self.dataset.hdf5_name}: {str(e)}")
			self.dataset.use_hdf5 = False

	# a very icky way to handle wildcard expansions
	def expand( self, path ):
		if not isinstance( path, Path ):
			path = Path(path)

		# do not glob if no wildcard to glob
		if "*" not in str(path):
			return [ path ]

		dir = path.parent
		name = path.name
		
		metadata_parent = cfg.metadata_dir / dir
		data_parent = cfg.data_dir / dir
		
		res = []
		# grab any paths from metadata folder (since this is for HDF5)
		if metadata_parent.exists():
			res = [ path.parent / child.stem for child in Path(metadata_parent).glob(name) ]
			# return if found anything
			if res:
				return res
		# grab anything from the data folder (if no metadata exists)
		if data_parent.exists():
			res = [ path.parent / child.name for child in Path(data_parent).glob(name) ]
			# return if found anything
			if res:
				return res
		
		# return an empty list
		if self.silent_errors:
			return []

		# raise an error to avoid headaches
		raise Exception(f'Cannot unglob requested path: {path}')


	def format( self, training=True ):
		if isinstance(self.dataset, type):
			self.dataset = dict()

		if isinstance(self.models, type):
			self.models = dict()

		if isinstance(self.loras, type):
			self.loras = dict()
		
		if isinstance(self.hyperparameters, type):
			self.hyperparameters = dict()
		
		if isinstance(self.evaluation, type):
			self.evaluation = dict()
		
		if isinstance(self.trainer, type):
			self.trainer = dict()
		
		if isinstance(self.inference, type):
			self.inference = dict()
		
		if isinstance(self.optimizations, type):
			self.optimizations = dict()

		if isinstance( self.dataset, dict ):
			self.dataset = Dataset(**self.dataset)

		if isinstance( self.hyperparameters, dict ):
			self.hyperparameters = Hyperparameters(**self.hyperparameters)

		if isinstance( self.evaluation, dict ):
			self.evaluation = Evaluation(**self.evaluation)

		if isinstance( self.trainer, dict ):
			self.trainer = Trainer(**self.trainer)

		if isinstance( self.trainer.deepspeed, dict ):
			self.trainer.deepspeed = DeepSpeed(**self.trainer.deepspeed)

		if isinstance( self.inference, dict ):
			self.inference = Inference(**self.inference)
		
		if isinstance( self.optimizations, dict ):
			self.optimizations = Optimizations(**self.optimizations)

		# convert to expanded paths
		self.dataset.training = [ self.expand(dir) for dir in self.dataset.training ]
		self.dataset.validation = [ self.expand(dir) for dir in self.dataset.validation ]
		self.dataset.noise = [ self.expand(dir) for dir in self.dataset.noise ]
		# flatten
		self.dataset.training = list(itertools.chain.from_iterable(self.dataset.training))
		self.dataset.validation = list(itertools.chain.from_iterable(self.dataset.validation))
		self.dataset.noise = list(itertools.chain.from_iterable(self.dataset.noise))

		# do cleanup
		for model in self.models:
			if not isinstance( model, dict ):
				continue

			# to-do: prune unused keys in here too automatically
			if "experimental" not in model or not model["experimental"]:
				model["experimental"] = {}

			if "prom_levels" in model:
				_logger.warning(f"Deprecated flag found: {'cfg.model.prom_levels'}")
				del model["prom_levels"]
			
			if "interleave" in model:
				_logger.warning(f"Deprecated flag found: {'cfg.model.interleave'}")
				del model["interleave"]

			if "p_rvq_levels" in model["experimental"]:
				model["experimental"]["rvq_levels_p"] = model["experimental"]["p_rvq_levels"]
				del model["experimental"]["p_rvq_levels"]
			
			if "p_len_train" in model["experimental"]:
				del model["experimental"]["p_len_train"]
			
			if "masking_ratio_fixed" in model["experimental"]:
				del model["experimental"]["masking_ratio_fixed"]

		self.models = [ Model(**model) if isinstance(model, dict) else model for model in self.models ]
		self.loras = [ LoRA(**lora)  if isinstance(lora, dict) else lora for lora in self.loras ]

		if not self.models:
			self.models = [ Model() ]

		for model in self.models:
			if isinstance( model.experimental, dict ):
				model.experimental = ModelExperimentalSettings(**model.experimental)

			if model.teacher:
				model.training = False
			if model.training:
				model.teacher = False

		if self.hyperparameters.scheduler_type and not self.hyperparameters.scheduler:
			self.hyperparameters.scheduler = self.hyperparameters.scheduler_type
			self.hyperparameters.scheduler_type = ""

		# do not combine the two
		if self.hyperparameters.scheduler == "schedulefree" and self.optimizations.dadaptation:
			self.hyperparameters.scheduler = ""

		if self.hyperparameters.scheduler == "":
			self.hyperparameters.torch_scheduler = True

		if self.trainer.backend == "local" and self.distributed:
			self.trainer.ddp = True
		
		if self.trainer.activation_checkpointing is not None:
			self.trainer.gradient_checkpointing = self.trainer.activation_checkpointing

		if not training:
			self.dataset.use_hdf5 = False

		# load our HDF5 file if requested here
		if self.dataset.use_hdf5:
			self.load_hdf5()

		# load tokenizer
		if self.tokenizer == "naive":
			self.tokenizer = NaiveTokenizer()
		else:
			from transformers import PreTrainedTokenizerFast

			tokenizer_path = self.rel_path / self.tokenizer_path
			# deduce path if a local copy is not provided
			if not tokenizer_path.exists():
				tokenizer_path = Path("./data/") / self.tokenizer_path
			
			if not self.silent_errors and not tokenizer_path.exists():
				raise Exception(f'Tokenizer path not found: {tokenizer_path}')

			self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
		
		if self.tokenizer == "naive":
			...
		else:
			from transformers import PreTrainedTokenizerFast

			text_tokenizer_path = self.rel_path / self.text_tokenizer_path
			# deduce path if a local copy is not provided
			if not text_tokenizer_path.exists():
				text_tokenizer_path = Path("./data/") / self.text_tokenizer_path
			
			if not self.silent_errors and not text_tokenizer_path.exists():
				raise Exception(f'Tokenizer path not found: {text_tokenizer_path}')

			self.text_tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(text_tokenizer_path))


# Preserves the old behavior
class NaiveTokenizer:
	def get_vocab( self ):
		"""
		if cfg.dataset.use_hdf5 and 'symmap' in cfg.hdf5:
			return json.loads( cfg.hdf5['symmap'].asstr()[()] )
		"""
		return {'<s>': 1, '</s>': 2, ' ': 3, '.': 4, ',': 5, '!': 6, '?': 7, 'p': 7, 'iː': 8, 'ɚ': 9, 'ˌ': 10, 'dˌ': 11, 'mˌ': 12, 'd': 13, 'ɹ': 14, 'tˈ': 15, 'pˌ': 16, 'uː': 17, 'l': 18, 'æ': 19, 'ɛ': 20, 'ɪ': 21, 'j': 22, 'ʊ': 23, 't': 24, 'n': 25, 'v': 26, 'a': 27, 'o': 28, 'ŋ': 29, 'w': 30, 'ʌ': 31, 'hˈ': 32, 'ɡˈ': 33, 'ə': 34, 'θˈ': 35, 'dˈ': 36, 'wˌ': 37, 'h': 38, 'z': 39, 'k': 40, 'ð': 41, 'ɡˌ': 42, 'ˈ': 43, 'fˈ': 44, 'i': 45, 's': 46, 'ʃ': 47, 'wˈ': 48, 'ðˈ': 49, 'ɹˈ': 50, 'lˈ': 51, 'ɡ': 52, 'oː': 53, 'mˈ': 54, 'e': 55, 'ɑː': 56, 'nˈ': 57, 'm': 58, 'θˌ': 59, 'sˈ': 60, 'f': 61, 'ɔː': 62, 'hˌ': 63, 'b': 64, 'jˈ': 65, 'ɐ': 66, 'ʒˈ': 67, 'θ': 68, 'bˈ': 69, 'ɾ': 70, 'ɜː': 71, 'ʌˈ': 72, 'ʃˌ': 73, 'bˌ': 74, 'kˈ': 75, 'ɔ': 76, 'zˈ': 77, 'ᵻ': 78, 'kˌ': 79, 'vˈ': 80, 'fˌ': 81, 'ʒ': 82, 'ʃˈ': 83, 'ɹˌ': 84, 'tˌ': 85, 'pˈ': 86, 'ðˌ': 87, 'sˌ': 88, 'nˌ': 89, 'lˌ': 90, '̩': 91, 'ʔ': 92, 'vˌ': 93, 'ɪˈ': 94, '"': 95, 'ɪˌ': 96, 'ʒˌ': 97, 'uːˌ': 98, 'ʊˈ': 99, 'jˌ': 100, 'uːˈ': 101, 'iːˈ': 102, 'zˌ': 103, '.ˈ': 104, '…': 105, 'ŋˌ': 106, 'ɐˌ': 107, '—ˈ': 108, 'iˌ': 109, 'iːˌ': 110, 'ɛː': 111, ')': 112, ')ˈ': 113, '(': 114, 'u': 115, '-': 116, 'ɖˈ': 117, 'iˈ': 118, 'ʰˈ': 119, 'ɟˈ': 120, '̃': 121, 'eː': 122, 'ɾˈ': 123, 'r': 124, 'ʰ': 125, '-ˌ': 126, 'ɫ': 127, 'q': 128, '—': 129, 'ʊˌ': 130, 'aː': 131, 'cˈ': 132, '…ˈ': 133, 'c': 134, 'ɳ': 135, 'ɐˈ': 136, 'x': 137, 'ʔˌ': 138, '.ˌ': 139, 'ɑ': 140, '?ˈ': 141, '̩ˈ': 142, '"ˈ': 143, ',ˈ': 144, 'ŋˈ': 145, 'əˌ': 146, '!ˈ': 147, '"ˌ': 148, '?ˌ': 149, ',ˌ': 150, '—ˌ': 151, '̩ˌ': 152, 'əˈ': 153, '!ˌ': 154, 'ɬ': 155, 'ʲ': 156, '¡': 157, 'ɯ': 158, 'qˌ': 159, 'ʑ': 160, 'ʑˈ': 161, '¿': 162, 'ɑːˈ': 163, 'iːː': 164, 'ɛˈ': 165, '¡ˈ': 166, 'æˈ': 167, 'ç': 168, 'ɾˌ': 169, 'ᵻˈ': 170, 'xˈ': 171, 'ɔːˈ': 172, ';': 173, 'ɬˌ': 174, ':': 175, 'ʔˈ': 176, 'ɑːˌ': 177, 'ɬˈ': 178, '”': 179, '“': 180, '“ˈ': 181, '“ˌ': 182, ';ˈ': 183, ';ˌ': 184, ':ˈ': 185, '1': 186, 'rˈ': 187, 'qˈ': 188, 'ᵻˌ': 189, 'ä': 190, '̞ˌ': 191, '̞': 192, 'ũˌ': 193, 'ʑˌ': 194, 'ᵝ': 195, 'ɽ': 196, 'ʲˌ': 197, 'ᵝˌ': 198, 'ũ': 199, 'ũˈ': 200, 'äˌ': 201, 'ɕ': 202, 'ɕˌ': 203, 'ɽˌ': 204, 'çˌ': 205, '…ˌ': 206, '̞ˈ': 207, 'äˈ': 208, 'ɽˈ': 209, 'ɸˌ': 210, 'ɴ': 211, 'ɸˈ': 212, 'ɕˈ': 213, 'ɸ': 214, 'ᵝˈ': 215, 'ʲˈ': 216, 'ĩ': 217, 'çˈ': 218, 'ĩˌ': 219, 'oˌ': 220, 'eˈ': 221, 'ʍ': 222, 'eˌ': 223, 'uˌ': 224, 'ʍˌ': 225, 'uˈ': 226, 'oˈ': 227, 'aˈ': 228}

	@cached_property
	def _bos_token( self ):
		return self.get_vocab()["<s>"]
	
	@cached_property
	def _eos_token( self ):
		return self.get_vocab()["</s>"]

	def encode( self, s ):
		symmap = self.get_vocab()
		phones = " ".join( list(s) )

		# do merge
		for merge in [ "\u02C8", "\u02CC", "\u02D0" ]:
			phones = phones.replace( f' {merge}', merge )

		phones = phones.split(" ")
		# cleanup
		phones = [ p for i, p in enumerate(phones) if p not in [" "] or ( p in [" "] and p != phones[i-1] ) ]
		# add bos / eos
		phones = ["<s>"] + [ " " if not p else p for p in phones ] + ["</s>"]
		# tokenize
		return [*map(symmap.get, phones)]

	def decode( self, t ):
		s = ""
		symmap = self.get_vocab()
		reverse_symmap = {}
		for k, v in symmap.items():
			reverse_symmap[v] = k

		for i, token in enumerate( t ):
			s += reverse_symmap[token]

		return s


_logger = logging.getLogger(__name__)

cfg = Config.from_cli()

# some safety for remapping deprecated formats and re-coercing uninitialized properties into actual types
try:
	cfg.format()
except Exception as e:
	if not cfg.silent_errors:
		raise e # throw an error because I'm tired of silent errors messing things up for me
	_logger.error(f"Error while parsing config YAML: {str(e)}")

if __name__ == "__main__":
	print(cfg)
