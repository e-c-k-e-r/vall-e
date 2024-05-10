import copy
import diskcache
import h5py
import json
import os
import subprocess
import sys
import time

import torch

from dataclasses import asdict, dataclass, field

from functools import cached_property
from pathlib import Path

from omegaconf import OmegaConf

from .utils.distributed import world_size

# Yuck
from transformers import PreTrainedTokenizerFast

@dataclass()
class _Config:
	cfg_path: str | None = None

	@property
	def relpath(self):
		return Path(self.cfg_path)

	@property
	def cache_dir(self):
		return self.relpath / ".cache"

	@property
	def data_dir(self):
		return self.relpath / "data"
	
	@property
	def metadata_dir(self):
		return self.relpath / "metadata"

	@property
	def ckpt_dir(self):
		return self.relpath / "ckpt"

	@property
	def log_dir(self):
		return self.relpath / "logs" / str(self.start_time)

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

	@staticmethod
	def _is_cfg_argv(s):
		return "=" in s and "--" not in s

	@classmethod
	def from_yaml( cls, yaml_path ):
		return cls.from_cli( [f'yaml="{yaml_path}"'] )

	@classmethod
	def from_cli(cls, args=sys.argv):
		cli_cfg = OmegaConf.from_cli([s for s in args if cls._is_cfg_argv(s)])

		# Replace argv to ensure there are no omegaconf options, for compatibility with argparse.
		sys.argv = [s for s in sys.argv if not cls._is_cfg_argv(s)]

		if cli_cfg.get("help"):
			print(f"Configurable hyperparameters with their default values:")
			print(json.dumps(asdict(cls()), indent=2, default=str))
			exit()

		if "yaml" in cli_cfg:
			yaml_cfg = OmegaConf.load(cli_cfg.yaml)
			yaml_path = Path(cli_cfg.yaml).absolute()
			cfg_path = Path(*yaml_path.relative_to(Path.cwd()).parts[:-1])
			cfg_path = cfg_path.with_suffix("")
			cfg_path = f'./{cfg_path}'

			yaml_cfg.setdefault("cfg_path", cfg_path)
			cli_cfg.pop("yaml")
		else:
			yaml_cfg = {}
		merged = OmegaConf.merge(yaml_cfg, cli_cfg)
		return cls(**dict(merged))

	def __repr__(self):
		return str(self)

	def __str__(self):
		return self.dumps()

@dataclass()
class Dataset:
	training: list[Path] = field(default_factory=lambda: [])
	validation: list[Path] = field(default_factory=lambda: [])
	noise: list[Path] = field(default_factory=lambda: [])
	
	temp: list[Path] = field(default_factory=lambda: [])

	speaker_name_getter: str = "lambda p: f'{p.parts[-3]}_{p.parts[-2]}'"
	speaker_group_getter: str = "lambda p: f'{p.parts[-3]}'"

	speaker_languages: dict = field(default_factory=lambda: {}) # dict where keys are the language codes and values are the speaker groups
	
	hdf5_name: str = "data.h5"
	use_hdf5: bool = False
	use_metadata: bool = False
	hdf5_flag: str = "a"
	validate: bool = True
	workers: int = 8
	cache: bool = True

	phones_range: list[int] = field(default_factory=lambda: [4, 256])
	duration_range: list[float] = field(default_factory=lambda: [1.0, 12.0])
	min_utterances: int = 2

	random_utterance: float = 1.0
	max_prompts: int = 3
	prompt_duration: float = 3.0
	
	max_resps: int = 1
	p_resp_append: float = 1.0

	sample_type: str = "path" # path | speaker
	
	tasks_list: list[str] = field(default_factory=lambda: ["tts"])
	
	_frames_per_second: int = 0 # allows setting your own hint

	@cached_property
	def frames_per_second(self):
		if self._frames_per_second > 0:
			return self._frames_per_second

		if cfg.inference.audio_backend == "dac":
			# using the 44KHz model with 24KHz sources has a frame rate of 41Hz
			if cfg.variable_sample_rate and cfg.sample_rate == 24_000:
				return 41
			if cfg.sample_rate == 44_000:
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

@dataclass()
class Model:
	_max_levels: int = 0
	_embeddings: str | None = None

	name: str = "" # vanity name for the model
	version: int = 1 # 1 = old with MultiEmbedding, 2 = new with AudioEmbedding
	size: str | dict = "full" # preset string or explicitly defined dimensionality
	resp_levels: int = 1 # RVQ-bin levels this model targets for outputs
	prom_levels: int = 8 # RVQ-bin levels this model accepts as an input prompt
	tasks: int = 8 # ["tts", "ns", "sr", "tse", "cse", "nse"] and leaves two more for anything else I want (like "svc")
	langs: int = 1 # defined languages
	tones: int = 1 # defined tones
	experts: int = 1
	arch_type: str = "retnet" # or "transformer""
	training: bool = True # unneeded now
	interleave: bool = False # use an interleaved AR rather than a split AR + NAR (experimental, worse performance and results)
	p_ar_level: float | str = "auto" # determines odds of selecting the AR (level 0) when training, "auto" for default behavior
	frozen_params: list[str] = field(default_factory=lambda: []) # frozen parameters that are not updated when training
	attention: str = "eager" # or flash_attention_2
	audio_embedding_sums: bool = True

	def get(self, name=None):
		return [ self ] if not name or self.name == name else []

	@property
	def max_levels(self):
		return self._max_levels if self._max_levels > 0 else self.prom_levels

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

		if self.arch_type != "transformer":
			if self.experts > 1:
				name.append(f'{self.experts}x'+self.arch_type.replace("/", "-"))
			else:
				name.append(self.arch_type.replace("/", "-"))

		if cfg.optimizations.bitnet:
			name.append("bitnet")

		if self.interleave:
			name.append("interleaved")
		else:
			name.append(f'{cfg.model.prom_levels}')


		return "-".join(name)

	@property
	def tokens(self):
		if isinstance(self.size, dict) and hasattr(self.size, "tokens"):
			return self.size['tokens']

		return 1024

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
	
@dataclass()
class Hyperparameters:
	batch_size: int = 8
	gradient_accumulation_steps: int = 32
	gradient_clipping: int | float = 100

	optimizer: str = "Adamw"
	optimizer_params: dict = field(default_factory=lambda: {}) # to pass through deepspeed config
	
	learning_rate: float = 3.25e-4
	warmup_steps: int = 0

	scheduler: str = ""
	scheduler_type: str = "" # deprecated
	scheduler_params: dict = field(default_factory=lambda: {}) # to pass through deepspeed config

	autotune: bool = False
	autotune_params: dict = field(default_factory=lambda: {}) # to pass through deepspeed config
	
	torch_optimizer: bool = False
	torch_scheduler: bool = False
	
@dataclass()
class Evaluation:
	batch_size: int = 64
	frequency: int = 250
	size: int = 64
  
	steps: int = 500
	ar_temperature: float = 1.0
	nar_temperature: float = 0.2

	load_disabled_engines: bool = True

@dataclass()
class DeepSpeed:
	zero_optimization_level: int = 0
	use_compression_training: bool = False
	compression_bits: int = 8
	inferencing: bool = False

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

		# documentation says neither can work
		if cfg.trainer.weight_dtype.lower() == "float16":
			cfg.trainer.amp = False

		autotune_params = cfg.hyperparameters.autotune_params

		if "enabled" not in autotune_params:
			autotune_params['enabled'] = True
		
		if "results_dir" not in autotune_params:
			autotune_params['results_dir'] = str( cfg.relpath / "autotune" / "results" )
		
		if "exps_dir" not in autotune_params:
			autotune_params['exps_dir'] = str( cfg.relpath / "autotune" / "exps_" )

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
				"auto_cast": False, # ???
			},
			"bf16": {
				"enabled": cfg.trainer.weight_dtype.lower() == "bfloat16",
			},
			"amp": {
				"enabled": cfg.trainer.amp,
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

		# disable local AMP
		cfg.trainer.amp = False

		null_keys = [ k for k in ds_cfg if not ds_cfg[k] ]
		for k in null_keys:
			del ds_cfg[k]

		if os.path.exists("./data/ds_config.json"):
			ds_cfg.update(json.load(open("./data/ds_config.json", "r", encoding="utf-8")))

		return ds_cfg

@dataclass()
class Trainer:
	iterations: int = 100_000

	save_tag: str = "step"
	load_tag: str | None = None

	save_on_oom: bool = True
	save_on_quit: bool = True
	
	export_on_save: bool = False
	export_on_quit: bool = False
	
	save_frequency: int = 100

	keep_last_checkpoints: int = 0

	load_state_dict: bool = False
	load_states: bool = True
	strict_loading: bool = True
	load_module_only: bool = False
	restart_step_count: bool = False

	activation_checkpointing: bool = True

	aggressive_optimizations: bool = False
	check_for_oom: bool = True
	gc_mode: str | None = None
	load_disabled_engines: bool = False

	weight_dtype: str = "float16"
	amp: bool = False
	ddp: bool = False

	load_webui: bool = False
	no_logger: bool = False

	backend: str = "local"
	deepspeed: DeepSpeed = field(default_factory=lambda: DeepSpeed)

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


@dataclass()
class Inference:
	backend: str = "local"
	weight_dtype: str = "float32"
	amp: bool = False

	normalize: bool = False # do NOT enable this unless you know exactly what you're doing
	audio_backend: str = "dac"

	# legacy / backwards compat
	use_vocos: bool = True
	use_encodec: bool = True
	use_dac: bool = True

	# shit that doesn't work
	recurrent_chunk_size: int = 0
	recurrent_forward: bool = False

	@cached_property
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

	linear: bool = True # inject/replace linear for BnB
	embedding: bool = True # inject/replace embedding for BnB
	optimizers: bool = True # inject/replace optimizers (BnB, DAdaptation)
	
	bitsandbytes: bool = False # use bitsandbytes
	dadaptation: bool = True # use dadaptation optimizer
	bitnet: bool = False # use bitnet
	fp8: bool = False # use fp8

@dataclass()
class Config(_Config):
	device: str = "cuda"
	mode: str = "training" # "inferencing"
	experimental: bool = False # So I can stop commenting out things when committing

	dataset: Dataset = field(default_factory=lambda: Dataset)
	model: Model = field(default_factory=lambda: Model)
	models: dict | list | None = None # deprecated
	hyperparameters: Hyperparameters = field(default_factory=lambda: Hyperparameters)
	evaluation: Evaluation = field(default_factory=lambda: Evaluation)
	trainer: Trainer = field(default_factory=lambda: Trainer)
	inference: Inference = field(default_factory=lambda: Inference)
	bitsandbytes: dict | list | None = None # deprecated
	optimizations: Optimizations = field(default_factory=lambda: Optimizations)
	
	tokenizer: str = "./tokenizer.json"

	sample_rate: int = 24_000
	variable_sample_rate: bool = False # NOT recommended, as running directly 24Khz audio in the 44Khz DAC model will have detrimental quality loss

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
		if self.cfg_path is not None and self.dataset.cache:
			return diskcache.Cache(self.cache_dir).memoize
		return lambda: lambda x: x

	def load_yaml( self, config_path ):
		tmp = Config.from_yaml( config_path )
		self.__dict__.update(tmp.__dict__)

	def load_hdf5( self, write=False ):
		if hasattr(self, 'hdf5'):
			self.hdf5.close()

		if self.distributed:
			self.dataset.hdf5_flag = "r"
		try:
			self.hdf5 = h5py.File(f'{self.cfg_path}/{self.dataset.hdf5_name}', 'a' if write else self.dataset.hdf5_flag) # to-do, have an easy to set flag that determines if training or creating the dataset
		except Exception as e:
			print("Error while opening HDF5 file:", f'{self.cfg_path}/{self.dataset.hdf5_name}', str(e))
			self.dataset.use_hdf5 = False

	def format( self ):
		self.dataset = Dataset(**self.dataset)
		self.dataset.training = [ Path(dir) for dir in self.dataset.training ]
		self.dataset.validation = [ Path(dir) for dir in self.dataset.validation ]
		self.dataset.noise = [ Path(dir) for dir in self.dataset.noise ]

		if self.models is not None:
			self.model = Model(**next(iter(self.models)))
		else:
			self.model = Model(**self.model)

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

# Preserves the old behavior
class NaiveTokenizer:
	def get_vocab( self ):
		"""
		if cfg.dataset.use_hdf5 and 'symmap' in cfg.hdf5:
			return json.loads( cfg.hdf5['symmap'].asstr()[()] )
		"""
		return {'<s>': 1, '</s>': 2, ' ': 3, '.': 4, ',': 5, '!': 6, '?': 7, 'p': 7, 'iː': 8, 'ɚ': 9, 'ˌ': 10, 'dˌ': 11, 'mˌ': 12, 'd': 13, 'ɹ': 14, 'tˈ': 15, 'pˌ': 16, 'uː': 17, 'l': 18, 'æ': 19, 'ɛ': 20, 'ɪ': 21, 'j': 22, 'ʊ': 23, 't': 24, 'n': 25, 'v': 26, 'a': 27, 'o': 28, 'ŋ': 29, 'w': 30, 'ʌ': 31, 'hˈ': 32, 'ɡˈ': 33, 'ə': 34, 'θˈ': 35, 'dˈ': 36, 'wˌ': 37, 'h': 38, 'z': 39, 'k': 40, 'ð': 41, 'ɡˌ': 42, 'ˈ': 43, 'fˈ': 44, 'i': 45, 's': 46, 'ʃ': 47, 'wˈ': 48, 'ðˈ': 49, 'ɹˈ': 50, 'lˈ': 51, 'ɡ': 52, 'oː': 53, 'mˈ': 54, 'e': 55, 'ɑː': 56, 'nˈ': 57, 'm': 58, 'θˌ': 59, 'sˈ': 60, 'f': 61, 'ɔː': 62, 'hˌ': 63, 'b': 64, 'jˈ': 65, 'ɐ': 66, 'ʒˈ': 67, 'θ': 68, 'bˈ': 69, 'ɾ': 70, 'ɜː': 71, 'ʌˈ': 72, 'ʃˌ': 73, 'bˌ': 74, 'kˈ': 75, 'ɔ': 76, 'zˈ': 77, 'ᵻ': 78, 'kˌ': 79, 'vˈ': 80, 'fˌ': 81, 'ʒ': 82, 'ʃˈ': 83, 'ɹˌ': 84, 'tˌ': 85, 'pˈ': 86, 'ðˌ': 87, 'sˌ': 88, 'nˌ': 89, 'lˌ': 90, '̩': 91, 'ʔ': 92, 'vˌ': 93, 'ɪˈ': 94, '"': 95, 'ɪˌ': 96, 'ʒˌ': 97, 'uːˌ': 98, 'ʊˈ': 99, 'jˌ': 100, 'uːˈ': 101, 'iːˈ': 102, 'zˌ': 103, '.ˈ': 104, '…': 105, 'ŋˌ': 106, 'ɐˌ': 107, '—ˈ': 108, 'iˌ': 109, 'iːˌ': 110, 'ɛː': 111, ')': 112, ')ˈ': 113, '(': 114, 'u': 115, '-': 116, 'ɖˈ': 117, 'iˈ': 118, 'ʰˈ': 119, 'ɟˈ': 120, '̃': 121, 'eː': 122, 'ɾˈ': 123, 'r': 124, 'ʰ': 125, '-ˌ': 126, 'ɫ': 127, 'q': 128, '—': 129, 'ʊˌ': 130, 'aː': 131, 'cˈ': 132, '…ˈ': 133, 'c': 134, 'ɳ': 135, 'ɐˈ': 136, 'x': 137, 'ʔˌ': 138, '.ˌ': 139, 'ɑ': 140, '?ˈ': 141, '̩ˈ': 142, '"ˈ': 143, ',ˈ': 144, 'ŋˈ': 145, 'əˌ': 146, '!ˈ': 147, '"ˌ': 148, '?ˌ': 149, ',ˌ': 150, '—ˌ': 151, '̩ˌ': 152, 'əˈ': 153, '!ˌ': 154, 'ɬ': 155, 'ʲ': 156, '¡': 157, 'ɯ': 158, 'qˌ': 159, 'ʑ': 160, 'ʑˈ': 161, '¿': 162, 'ɑːˈ': 163, 'iːː': 164, 'ɛˈ': 165, '¡ˈ': 166, 'æˈ': 167, 'ç': 168, 'ɾˌ': 169, 'ᵻˈ': 170, 'xˈ': 171, 'ɔːˈ': 172, ';': 173, 'ɬˌ': 174, ':': 175, 'ʔˈ': 176, 'ɑːˌ': 177, 'ɬˈ': 178, '”': 179, '“': 180, '“ˈ': 181, '“ˌ': 182, ';ˈ': 183, ';ˌ': 184, ':ˈ': 185, '1': 186, 'rˈ': 187, 'qˈ': 188, 'ᵻˌ': 189, 'ä': 190, '̞ˌ': 191, '̞': 192, 'ũˌ': 193, 'ʑˌ': 194, 'ᵝ': 195, 'ɽ': 196, 'ʲˌ': 197, 'ᵝˌ': 198, 'ũ': 199, 'ũˈ': 200, 'äˌ': 201, 'ɕ': 202, 'ɕˌ': 203, 'ɽˌ': 204, 'çˌ': 205, '…ˌ': 206, '̞ˈ': 207, 'äˈ': 208, 'ɽˈ': 209, 'ɸˌ': 210, 'ɴ': 211, 'ɸˈ': 212, 'ɕˈ': 213, 'ɸ': 214, 'ᵝˈ': 215, 'ʲˈ': 216, 'ĩ': 217, 'çˈ': 218, 'ĩˌ': 219, 'oˌ': 220, 'eˈ': 221, 'ʍ': 222, 'eˌ': 223, 'uˌ': 224, 'ʍˌ': 225, 'uˈ': 226, 'oˈ': 227, 'aˈ': 228}

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

# OmegaConf might not coerce the dicts into the @dataclass decorated classes, so we (try to) coerce them ourselves
try:
	cfg.format()
	if cfg.dataset.use_hdf5:
		cfg.load_hdf5()
except Exception as e:
	print("Error while parsing config YAML:", e)
	pass

try:
	from transformers import PreTrainedTokenizerFast
	cfg.tokenizer = (cfg.relpath if cfg.cfg_path is not None else Path("./data/")) / cfg.tokenizer
	cfg.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(cfg.tokenizer))
except Exception as e:
	cfg.tokenizer = NaiveTokenizer()
	print("Error while parsing tokenizer:", e)
	pass

if __name__ == "__main__":
	print(cfg)
