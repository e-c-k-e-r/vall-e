import copy
import diskcache
import h5py
import json
import os
import subprocess
import sys
import time

import torch

from dataclasses import asdict, dataclass
from dataclasses import dataclass, field

from functools import cached_property
from pathlib import Path

from omegaconf import OmegaConf

from .utils.distributed import world_size

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
	name: str = "" # vanity name for the model
	version: int = 1 # 1 = old with MultiEmbedding, 2 = new with AudioEmbedding
	size: str | dict = "full" # preset string or explicitly defined dimensionality
	resp_levels: int = 1 # RVQ-bin levels this model targets for outputs
	prom_levels: int = 8 # RVQ-bin levels this model accepts as an input prompt
	tasks: int = 8 # ["tts", "ns", "sr", "tse", "cse", "nse"] and leaves two more for anything else I want (like "svc")
	langs: int = 1 # defined languages
	arch_type: str = "retnet" # or "transformer""
	training: bool = True # unneeded now
	interleave: bool = False # use an interleaved AR rather than a split AR + NAR (experimental, worse performance and results)
	p_ar_level: float | str = "auto" # determines odds of selecting the AR (level 0) when training, "auto" for default behavior
	frozen_params: list[str] = field(default_factory=lambda: []) # frozen parameters that are not updated when training

	@property
	def full_name(self):
		name = [ self.name ]
		
		if self.size != "full" and isinstance(self.size, str):
			name.append(self.size)

		if self.arch_type != "transformer":
			name.append(self.arch_type.replace("/", "-"))

		if self.interleave:
			name.append("interleaved")

		name.append(f'{cfg.models.prom_levels}')

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
class Models:
	_max_levels: int = 0
	_prom_levels: int = 1

	_models: list[Model] = field(default_factory=lambda: [
		Model(name="ar", resp_levels=1, prom_levels=8, tasks=8, langs=1, training=True, interleave=False),
		Model(name="nar", resp_levels=7, prom_levels=8, tasks=8, langs=1, training=True, interleave=False),
	])

	def get(self, name=None):
		if not name:
			return [ Model(**model) for model in self._models ]

		for model in self._models:
			if model.name == name:
				return model

		raise ValueError

	@property
	def ar(self):
		return self.get("ar")

	@property
	def ar_nar(self):
		return self.get("ar+nar")

	@property
	def nar(self):
		return self.get("nar")

	@property
	def prom_levels(self):
		prom_levels = self._prom_levels
		for model in self._models:
			prom_levels = max(prom_levels, model.prom_levels)
		return prom_levels

	@property
	def tasks(self):
		tasks = 1
		for model in self._models:
			tasks = max(tasks, model.tasks)
		return tasks

	@property
	def max_levels(self):
		return self._max_levels if self._max_levels > 0 else self.prom_levels
	
@dataclass()
class Hyperparameters:
	batch_size: int = 8
	gradient_accumulation_steps: int = 32
	gradient_clipping: int = 100

	optimizer: str = "Adamw"
	torch_optimizer: bool = False
	optimizer_params: dict = field(default_factory=lambda: {})
	learning_rate: float = 3.25e-4

	scheduler_type: str = ""
	scheduler_params: dict = field(default_factory=lambda: {})
	
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
		scheduler_params = {}
		for k in cfg.hyperparameters.scheduler_params:
			scheduler_params[k] = cfg.hyperparameters.scheduler_params[k]

		if cfg.hyperparameters.scheduler_type == "WarmupDecayLR" and 'total_num_steps' not in scheduler_params:
			scheduler_params['total_num_steps'] = cfg.trainer.iterations

		ds_cfg = {
			"train_micro_batch_size_per_gpu": cfg.hyperparameters.batch_size,
			"gradient_accumulation_steps": cfg.hyperparameters.gradient_accumulation_steps,
			"optimizer": {
				"type": cfg.hyperparameters.optimizer,
				"params": {
					"lr": cfg.hyperparameters.learning_rate,
				}
			} if not cfg.hyperparameters.torch_optimizer else None,
			"scheduler": {
				"type": cfg.hyperparameters.scheduler_type,
				"params": scheduler_params,
			} if cfg.hyperparameters.scheduler_type != "" else None,
			"gradient_clipping": cfg.hyperparameters.gradient_clipping,
			"fp16": {
				"enabled": True,
				"auto_cast": True,
			} if cfg.trainer.weight_dtype.lower() == "float16" and not cfg.trainer.amp else None,
			"bf16": {
				"enabled": cfg.trainer.weight_dtype.lower() == "bfloat16" and not cfg.trainer.amp
			},
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
							"modules": [
							#	"^.+?$"
								"blocks", # for transformer-based models
								"retnet", # for RetNets-based models
							]
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
							"modules": [
							#	"^.+?$"
								"blocks", # for transformer-based models
								"retnet", # for RetNets-based models
							]
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

	load_webui: bool = False

	backend: str = "local"
	deepspeed: DeepSpeed = field(default_factory=lambda: DeepSpeed)

	@cached_property
	def dtype(self):
		if self.weight_dtype == "float16":
			return torch.float16
		if self.weight_dtype == "bfloat16":
			return torch.bfloat16
		return torch.float32


@dataclass()
class Inference:
	backend: str = "local"
	weight_dtype: str = "float32"
	amp: bool = False

	normalize: bool = False # do NOT enable this unless you know exactly what you're doing
	use_vocos: bool = True

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
		return torch.float32

@dataclass()
class BitsAndBytes:
	enabled: bool = False
	injects: bool = False

	linear: bool = True
	embedding: bool = True

@dataclass()
class Config(_Config):
	device: str = "cuda"
	mode: str = "training" # "inferencing"
	experimental: bool = False # So I can stop commenting out things when committing

	dataset: Dataset = field(default_factory=lambda: Dataset)
	models: Models = field(default_factory=lambda: Models)
	hyperparameters: Hyperparameters = field(default_factory=lambda: Hyperparameters)
	evaluation: Evaluation = field(default_factory=lambda: Evaluation)
	trainer: Trainer = field(default_factory=lambda: Trainer)
	inference: Inference = field(default_factory=lambda: Inference)
	bitsandbytes: BitsAndBytes = field(default_factory=lambda: BitsAndBytes)

	@property
	def sample_rate(self):
		return 24_000

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
		self.models = Models(**self.models)
		self.hyperparameters = Hyperparameters(**self.hyperparameters)
		self.evaluation = Evaluation(**self.evaluation)
		self.trainer = Trainer(**self.trainer)
		self.inference = Inference(**self.inference)
		self.bitsandbytes = BitsAndBytes(**self.bitsandbytes)

		self.trainer.deepspeed = DeepSpeed(**self.trainer.deepspeed)
	
		self.dataset.training = [ Path(dir) for dir in self.dataset.training ]
		self.dataset.validation = [ Path(dir) for dir in self.dataset.validation ]
		self.dataset.noise = [ Path(dir) for dir in self.dataset.noise ]


cfg = Config.from_cli()

# OmegaConf might not coerce the dicts into the @dataclass decorated classes, so we (try to) coerce them ourselves
try:
	cfg.format()

	# cached_property stopped working...
	if cfg.dataset.use_hdf5:
		cfg.load_hdf5()


except Exception as e:
	pass


if __name__ == "__main__":
	print(cfg)
