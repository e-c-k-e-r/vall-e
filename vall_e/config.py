import copy
import diskcache
import h5py
import json
import os
import subprocess
import sys
import time
import argparse
import yaml
import random

import torch
import numpy as np

from dataclasses import asdict, dataclass, field

from functools import cached_property
from pathlib import Path

from .utils.distributed import world_size


def set_seed(seed=None):
	if not seed:
		seed = time.time()

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

@dataclass()
class BaseConfig:
	yaml_path: str | None = None # path passed in through --yaml

	@property
	def cfg_path(self):
		return Path(self.yaml_path.parent) if self.yaml_path is not None else None

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

	@classmethod
	def from_yaml( cls, yaml_path ):
		state = {}
		state = yaml.safe_load(open(yaml_path, "r", encoding="utf-8"))
		state.setdefault("yaml_path", yaml_path)
		return cls(**state)

	@classmethod
	def from_cli(cls, args=sys.argv):
		# legacy support for yaml=`` format
		for i, arg in enumerate(args):
			if arg.startswith("yaml"):
				args[i] = f'--{arg}'

		parser = argparse.ArgumentParser(allow_abbrev=False)
		parser.add_argument("--yaml", type=Path, default=os.environ.get('VALLE_YAML', None)) # os environ so it can be specified in a HuggingFace Space too
		args, unknown = parser.parse_known_args(args=args)

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
	
	temp: list[Path] = field(default_factory=lambda: []) # for when I need to yank things out of a training dataset without cutting it out

	speaker_name_getter: str = "lambda p: f'{p.parts[-3]}_{p.parts[-2]}'" # function eval'd to extract a speaker's name from an utternace path
	speaker_group_getter: str = "lambda p: f'{p.parts[-3]}'" # function eval'd to extract a speaker's group from an utternace path

	speaker_languages: dict = field(default_factory=lambda: {}) # dict where keys are the language codes and values are the speaker groups
	
	hdf5_name: str = "data.h5" # file name to load the HDF5 dataset
	use_hdf5: bool = False # whether to load from an HDF5 dataset
	hdf5_flag: str = "a" # flag to load the HDF5 file, automatically adjusted anyways
	use_metadata: bool = False # use genretaed metadata to aid in dataset loading
	
	validate: bool = True # validate each utterance on wheter it can be included based on duration range caps
	workers: int = 8 # number of dataloader workers to spawn
	cache: bool = True # use diskcache to cache the dataset

	phones_range: list[int] = field(default_factory=lambda: [4, 256]) # deprecated, the amount of phonemes an utterance can be to be included in the dataset
	duration_range: list[float] = field(default_factory=lambda: [1.0, 12.0]) # the duration range an utterance can be to be included in the dataset
	prompt_duration_range: list[float] = field(default_factory=lambda: [3.0, 6.0]) # the duration range the input prompts can be
	min_utterances: int = 2 # minimum number of utterances a speaker can have

	random_utterance: float = 1.0 # probability to use a different utterance rather than using the target utterance as an input prompt
	max_prompts: int = 3 # maximum number of utterances that can be included in an input prompt for training
	
	prompt_duration: float | None = None # legacy
	
	max_resps: int = 1 # number of samples to target for training
	p_resp_append: float = 1.0 # probability to append another sample to the training target

	sample_type: str = "path" # path | speaker
	sample_order: str = "interleaved" # duration
	sample_max_duration_batch: float = 0.0 # total number of seconds of utterances per batched, 0 to disable
	# for a full sized model with 12GiB of VRAM for Encodec, 120 seconds is just enough
	sample_shuffle: bool = True # 

	tasks_list: list[str] = field(default_factory=lambda: ["tts"]) # list of tasks to train against
	reencode_on_concat: bool = False # whether to concat audio by decode => concat => encode, or naively concat codes
	reencode_device: str = "cpu" # "cpu" is slower but saves memory, cuda throws [rank0]: RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
	noise_scale: float = 0.25 # scaling noise value
	inject_noise_in_prom: bool = False # adds noise to the input prompt waveform to try and vary things
	
	_frames_per_second: int = 0 # allows setting your own hint

	@cached_property
	def frames_per_second(self):
		if self._frames_per_second > 0:
			return self._frames_per_second

		if cfg.audio_backend == "dac":
			# using the 44KHz model with 24KHz sources has a frame rate of 41Hz
			if cfg.variable_sample_rate and cfg.sample_rate == 24_000:
				return 41
			if cfg.sample_rate == 44_000 or cfg.sample_rate == 44_100: # to-do: find the actual value for 44.1K
				return 86
			if cfg.sample_rate == 16_000:
				return 50
		
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
	audio_embedding_mode: str | None = None # None | "exclusive" | "inclusive", subjugates the audio backend's encoding/decoding model for embeddings
	kv_heads: int = 0 # MHA or GQA (for supported backends)
	p_rvq_levels: str | list = "auto" # determines odds of selecting RVQ levels when training, "equal" will make each level equally likely
	rvq_level_range: list = field(default_factory=lambda: []) # some cringe to try and limit the RVQ training range for LoRAs, isn't necesary
	unified_position_ids: bool = True # False will generate position IDs partitioned for each section
	tie_classifier_to_embedding: bool = False # Ties the classifier output to their respective embeddings, this does not seem to do anything good in testing
	
	# performs token dropout to compensate for errors
	token_dropout_error: float = 0.0 # probability to nudge a token by ±1
	token_dropout_rate: float = 0.0 # probability to randomly set a token to a special dropout value
	token_dropout_rvq_levels: list = field(default_factory=lambda: [1,8]) # determines which levels to do dropout, by default do not do dropout on RVQ level 0

	causal_size: int = 1 # experimental setting to see if I can just do parallel decoding in chunks instead of one-at-a-time without resorting to exotic solutions
	# VALL-E 2's approach of "combining token embeddings to group them" sounds terribad for a shared AR/NAR model
	# however, introducing partial parallel decoding for the AR maybe maybe MAYBE might help try and unify the AR/NAR tasks better, MAYBE
	# it just seems like a bitch to try and train something worthwhile with it, since there's crackles every other token

	p_len_train: float = 0.05 # odds of injecting a "len" task within the model for NAR-len
	# to-to: just incorporate this as a task instead

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
	training: bool = True # I really need to attend to this
	frozen_params: list[str] = field(default_factory=lambda: []) # frozen parameters that are not updated when training
	attention: str = "auto" # for llama arch_types: attention used
	dropout: float = 0.1 # adjustable dropout value
	#loss_factors: dict = field(default_factory=lambda: { "text": 0.1, "prom": 1.0, "resp": 1.0 }) # disable it by default since it causes a little more harm than good
	loss_factors: dict = field(default_factory=lambda: {})
	capabilities: list = field(default_factory=lambda: ["ar", "nar"]) # + ["lang", "tone"] if you have your dataset labeled for such
	
	experimental: dict | ModelExperimentalSettings | None = None # experimental settings

	def get(self, name=None):
		return [ self ] if not name or self.name == name else []
	
	def loss_factor(self, k):
		return self.loss_factors[k] if k in self.loss_factors else 1.0

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
		elif isinstance(self.size, str) and self.size != "full":
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
	gradient_clipping: int | float = 10 # largest size a gradient norm can be

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
	
@dataclass()
class Evaluation:
	batch_size: int = 64 # number of samples per batch during eval / val
	frequency: int = 250 # do eval / val every X iterations
	size: int = 64 # number of samples to generate during eval / val

	steps: int = 500
	ar_temperature: float = 1.0 # AR temp for inferencing
	nar_temperature: float = 0.0 # NAR temp for inferencing
	nar_levels: int = 0 # maximum NAR levels to use for inferencing

	load_disabled_engines: bool = True # see the other load_disabled_engines

@dataclass()
class DeepSpeed:
	zero_optimization_level: int = 0 # doesn't seem to work
	use_compression_training: bool = False # cope
	compression_bits: int = 8 # cope
	inferencing: bool = False # for using DeepSpeed's inferencing wrapper instead
	
	amp: bool = False # use DeepSpeed's AMP (requires some other package installed apparently)

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
			ds_cfg.update(json.load(open("./data/ds_config.json", "r", encoding="utf-8")))
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
	resize_modules: bool = False # automatically resizes 

	activation_checkpointing: bool | None = None # deprecated, should technically be used for only on activations and not the entire gradients, but HF only has gradient checkpointing
	gradient_checkpointing: bool = True # enables gradient checkpointing to save VRAM at the cost of slightly reduced performance when training

	aggressive_optimizations: bool = False # deprecated
	check_for_oom: bool = True # checks for OOMs thrown during forward/backwards
	gc_mode: str | None = None # deprecated, but marks when to do GC
	load_disabled_engines: bool = False # deprecated, but signals to load engines not used for training for, for example, evaluation/validation

	weight_dtype: str = "float16" # dtype to have the model under

	amp: bool = False # automatic mixed precision
	ddp: bool = False # torch's internal DDP, automatically set if local backend is used and multiple GPUs are requested

	load_webui: bool = False # not working, but loads the web UI to allow inferencing during training
	no_logger: bool = False # deprecated, but reroutes some logger calls to normal print statements for when logger broke because of BitNet

	backend: str = "local" # training backend to use. currently supports "local" | "deepspeed"
	deepspeed: DeepSpeed = field(default_factory=lambda: DeepSpeed) # deepspeed settings

	@cached_property
	def dtype(self):
		if self.weight_dtype == "float16":
			return torch.float16
		if self.weight_dtype == "bfloat16":
			return torch.bfloat16
		if self.weight_dtype == "float8_e5m2":
			return torch.float8_e5m2
		if self.weight_dtype == "float8_e4m3fn":
			return torch.float8_e4m3fn
		return torch.float32

	@cached_property
	def scale_loss(self):
		# currently cannot feasibly apply loss scaling with DeepSpeed backend (it can handle it itself anyways)
		if self.backend != "local":
			return False
		return self.dtype == torch.float16


@dataclass()
class Inference:
	backend: str = "local" # backend to use when inferencing
	weight_dtype: str = "float32" # dtype to load the model under
	amp: bool = False # automatic mixed precision during inferencing

	normalize: bool = False # do NOT enable this unless you know exactly what you're doing

	# legacy / backwards compat
	audio_backend: str = "" # encodec, vocos, dac
	use_vocos: bool = True
	use_encodec: bool = True
	use_dac: bool = True

	@property
	def dtype(self):
		if self.weight_dtype == "float16":
			return torch.float16
		if self.weight_dtype == "bfloat16":
			return torch.bfloat16
		if self.weight_dtype == "int8":
			return torch.int8
		if self.weight_dtype == "float8_e5m2":
			return torch.float8_e5m2
		if self.weight_dtype == "float8_e4m3fn":
			return torch.float8_e4m3fn
		return torch.float32

# should be renamed to optimizations
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

	model_offloading: dict | None = None # automatically splits the model over a list of devices
	# example: {"include":["model"], "limits": [ (6 * 1024) * (1024 ** 2), -1 ]} will have the GPU capped to 6GiB, and offload the remaining layers to CPU
	# example: {"include":["model"], "device": ["cuda:0", "cuda:1"], "limits": [ 0.5, 0.5 ]} will have the GPU 1 try and use 50% of the model, and GPU 2 try and use the other 50%
	# | {"assign": [[ f'layers.{i}.' for i in range(0,6) ], [ f'layers.{i}.' for i in range(6,12) ]]} will assign layers 0-5 to device 1, and 6-12 to device 2

	tensorrt: bool = False

@dataclass()
class Config(BaseConfig):
	device: str = "cuda" # target device
	mode: str = "training" # "inferencing"
	experimental: bool = False # Debug flag, unused now

	dataset: Dataset = field(default_factory=lambda: Dataset)
	models: dict | list | None = field(default_factory=lambda: [])
	loras: dict | list | None = field(default_factory=lambda: [])
	hyperparameters: Hyperparameters = field(default_factory=lambda: Hyperparameters)
	evaluation: Evaluation = field(default_factory=lambda: Evaluation)
	trainer: Trainer = field(default_factory=lambda: Trainer)
	inference: Inference = field(default_factory=lambda: Inference)
	bitsandbytes: dict | list | None = None # deprecated
	optimizations: Optimizations = field(default_factory=lambda: Optimizations)
	
	tokenizer: str | None = None # tokenizer class
	tokenizer_path: str = "./tokenizer.json" # tokenizer path

	sample_rate: int = 24_000 # sample rate the model expects
	variable_sample_rate: bool = False # NOT recommended, as running directly 24Khz audio in the 44Khz DAC model will have detrimental quality loss

	audio_backend: str = "vocos" # audio backend to use "encodec" | "vocos" | "dac""

	weights_format: str = "pth" # "pth" | "sft"

	@property
	def model(self):
		for i, model in enumerate(self.models):
			if model.training:
				return model

		return self.models[0] if len(self.models) > 0 else None

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

	@cached_property
	def diskcache(self):
		if self.yaml_path is not None and self.dataset.cache:
			return diskcache.Cache(self.cache_dir).memoize
		return lambda: lambda x: x

	# I don't remember why this is needed
	def load_yaml( self, config_path ):
		tmp = Config.from_yaml( config_path )
		self.__dict__.update(tmp.__dict__)

	def load_hdf5( self, write=False ):
		if hasattr(self, 'hdf5'):
			self.hdf5.close()

		if self.distributed:
			self.dataset.hdf5_flag = "r"
		try:
			self.hdf5 = h5py.File(f'{self.rel_path}/{self.dataset.hdf5_name}', 'a' if write else self.dataset.hdf5_flag) # to-do, have an easy to set flag that determines if training or creating the dataset
		except Exception as e:
			print("Error while opening HDF5 file:", f'{self.rel_path}/{self.dataset.hdf5_name}', str(e))
			self.dataset.use_hdf5 = False

	# to-do: prune unused keys
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

		self.dataset = Dataset(**self.dataset)
		self.dataset.training = [ Path(dir) for dir in self.dataset.training ]
		self.dataset.validation = [ Path(dir) for dir in self.dataset.validation ]
		self.dataset.noise = [ Path(dir) for dir in self.dataset.noise ]

		# do cleanup
		for model in self.models:
			if not isinstance( model, dict ):
				continue

			if "prom_levels" in model:
				del model["prom_levels"]
			
			if "interleave" in model:
				del model["interleave"]

			if "audio_embedding_sums" not in model:
				continue

			if "experimental" not in model or not model["experimental"]:
				model["experimental"] = {}

			model["experimental"]["audio_embedding_sums"] = model.pop("audio_embedding_sums")


		self.models = [ Model(**model) for model in self.models ]
		self.loras = [ LoRA(**lora) for lora in self.loras ]

		if not self.models:
			self.models = [ Model() ]

		for model in self.models:
			if not isinstance( model.experimental, dict ):
				continue
			model.experimental = ModelExperimentalSettings(**model.experimental)

		self.hyperparameters = Hyperparameters(**self.hyperparameters)

		self.evaluation = Evaluation(**self.evaluation)

		self.trainer = Trainer(**self.trainer)

		if not isinstance(self.trainer.deepspeed, type):
			self.trainer.deepspeed = DeepSpeed(**self.trainer.deepspeed)

		self.inference = Inference(**self.inference)

		if self.bitsandbytes is not None:
			self.optimizations = Optimizations(**self.bitsandbytes)
		else:
			self.optimizations = Optimizations(**self.optimizations)

		if self.hyperparameters.scheduler_type and not self.hyperparameters.scheduler:
			self.hyperparameters.scheduler = self.hyperparameters.scheduler_type
			self.hyperparameters.scheduler_type = ""

		# do not combine the two
		if self.hyperparameters.scheduler == "schedulefree" and self.optimizations.dadaptation:
			self.hyperparameters.scheduler = ""

		if self.hyperparameters.scheduler == "":
			self.hyperparameters.torch_scheduler = True

		if self.dataset.prompt_duration is not None:
			self.dataset.prompt_duration_range = [self.dataset.prompt_duration, self.dataset.prompt_duration]

		if self.trainer.backend == "local" and self.distributed:
			self.trainer.ddp = True
		
		if self.inference.audio_backend != "" and self.audio_backend == "":
			self.audio_backend = self.inference.audio_backend

		if self.trainer.activation_checkpointing is not None:
			self.trainer.gradient_checkpointing = self.trainer.activation_checkpointing

		if not training:
			self.dataset.use_hdf5 = False

		# load our HDF5 file if requested here
		if self.dataset.use_hdf5:
			self.load_hdf5()

		# load tokenizer
		if cfg.tokenizer == "naive":
			cfg.tokenizer = NaiveTokenizer()
		else:
			try:
				from transformers import PreTrainedTokenizerFast

				tokenizer_path = cfg.rel_path / cfg.tokenizer_path if cfg.yaml_path is not None else None
				if tokenizer_path and not tokenizer_path.exists():
					tokenizer_path = Path("./data/") / cfg.tokenizer_path
				
				if tokenizer_path and tokenizer_path.exists():
					cfg.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
				else:
					cfg.tokenizer = NaiveTokenizer()
			except Exception as e:
				cfg.tokenizer = NaiveTokenizer()
				print("Error while parsing tokenizer:", e)
				pass


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


cfg = Config.from_cli()

# some safety for remapping deprecated formats and re-coercing uninitialized properties into actual types
try:
	cfg.format()
except Exception as e:
	print("Error while parsing config YAML:")
	raise e # throw an error because I'm tired of silent errors messing things up for me

if __name__ == "__main__":
	print(cfg)
