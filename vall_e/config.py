import copy
import diskcache
import h5py
import json
import os
import subprocess
import sys
import time

from dataclasses import asdict, dataclass
from dataclasses import dataclass, field

from functools import cached_property
from pathlib import Path

from omegaconf import OmegaConf

@dataclass()
class _Config:
	cfg_path: str | None = None

	@property
	def relpath(self):
		return Path(self.cfg_path)

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
	
	temp: list[Path] = field(default_factory=lambda: [])

	speaker_name_getter: str = "lambda p: f'{p.parts[-3]}_{p.parts[-2]}'"
	
	hdf5_name: str = "data.h5"
	use_hdf5: bool = False
	validate: bool = True
	workers: int = 8
	cache: bool = True

	phones_range: list[int] = field(default_factory=lambda: [4, 256])
	duration_range: list[float] = field(default_factory=lambda: [1.0, 12.0])

	random_utterance: float = 1.0
	max_prompts: int = 3
	prompt_duration: float = 3.0

@dataclass()
class Model:
	name: str = ""
	size: str = "full"
	resp_levels: int = 1
	arch_type: str = "transformer"

	@property
	def scale(self):
		if self.size == "quarter":
			return 0.25
		if self.size == "half":
			return 0.5
		return 1.0

	@property
	def full_name(self):
		name = [ self.name ]
		
		if self.size != "full":
			name.append(self.size)

		if self.arch_type != "transformer":
			name.append(self.arch_type.replace("/", "-"))

		name.append(f'{cfg.models.levels}')

		return "-".join(name)

	@property
	def tokens(self):
		return 1024

	@property
	def dim(self):
		if self.size == "quarter":
			return 256
		if self.size == "half":
			return 512
		if self.size == "full":
			return 1024
		raise ValueError

	@property
	def heads(self):
		if self.size == "quarter":
			return 4
		if self.size == "half":
			return 8
		if self.size == "full":
			return 16
		raise ValueError

	@property
	def layers(self):
		return 12

@dataclass()
class Models:
	_models: list[Model] = field(default_factory=lambda: [
		Model(name="ar", resp_levels=1),
		Model(name="nar", resp_levels=7),
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
	def nar(self):
		return self.get("nar")

	@property
	def levels(self):
		return self.prom_levels
	
	prom_levels: int = 8

@dataclass()
class Hyperparameters:
	batch_size: int = 8
	gradient_accumulation_steps: int = 32
	gradient_clipping: int = 100

	optimizer: str = "Adamw"
	learning_rate: float = 3.25e-4

	scheduler_type: str = ""
	scheduler_params: dict = field(default_factory=lambda: {})
	
@dataclass()
class Evaluation:
	batch_size: int = 64
	frequency: int = 250
	size: int = 64
  
	steps: int = 500
	temperature: float = 1.0

@dataclass()
class Trainer:
	iterations: int = 100_000

	save_tag: str = "step"
	load_tag: str | None = None

	save_on_oom: bool = True
	save_on_quit: bool = True
	save_frequency: int = 100

	load_state_dict: bool = False
	load_states: bool = True
	strict_loading: bool = True
	restart_step_count: bool = False

	aggressive_optimizations: bool = False

	gc_mode: str | None = None

	weight_dtype: str = "float16"

	zero_optimization_level: int = 0
	use_compression_training: bool = False


@dataclass()
class Inference:
	use_vocos: bool = True

@dataclass()
class BitsAndBytes:
	enabled: bool = False
	injects: bool = False

	linear: bool = False
	embedding: bool = False

@dataclass()
class Config(_Config):
	device: str = "cuda"

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

	@cached_property
	def get_spkr(self):
		return eval(self.dataset.speaker_name_getter)

	@property
	def scheduler(self):
		cfg = {
			"type": self.hyperparameters.scheduler_type,
			"params": {},
		}

		for k in self.hyperparameters.scheduler_params:
			cfg['params'][k] = self.hyperparameters.scheduler_params[k]

		if self.hyperparameters.scheduler_type == "WarmupDecayLR" and 'total_num_steps' not in cfg['params']:
			cfg['params']['total_num_steps'] = self.trainer.iterations
		return cfg

	@property
	def fp16_cfg(self):
		if self.trainer.weight_dtype.lower() != "float16":
			return None
		return {
			"enabled": True,
			"auto_cast": True,
		}

	@property
	def bf16_cfg(self):
		return {
			"enabled": self.trainer.weight_dtype.lower() == "bfloat16"
		}
	
	def get_compression_cfg(self, model):
		if not self.trainer.use_compression_training:
			return None
		
		weights = [ name[0] for name in model.named_parameters() ]
		bits = 8

		return {
			"weight_quantization": {
				"shared_parameters":{
					"enabled": True,
					"quantizer_kernel": True,
					"schedule_offset": 0,
					"quantize_groups": 64,
					"quantize_verbose": True,
					"quantization_type": "symmetric",
					"rounding": "nearest",
					"quantize_weight_in_forward": True,
					"fp16_mixed_quantize":{
						"enabled": False,
						"quantize_change_ratio": 1
					}
				},
				"different_groups": {
					"wq1": {
						"params": {
							"start_bits": bits,
							"target_bits": bits,
							"quantization_period": 0
						},
						"modules": weights
					}
				}
			},
			"activation_quantization": {
				"shared_parameters":{
					"enabled": True,
					"quantization_type": "symmetric",
					"range_calibration": "dynamic",
					"schedule_offset": 0
				},
				"different_groups": {
					"aq1": {
						"params": {
							"bits": bits
						},
						"modules": weights
					}
				}
			}
		}

	@property
	def zero_cfg(self):
		if self.trainer.zero_optimization_level == 0:
			return None

		return {
			"stage": self.trainer.zero_optimization_level,
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
			}
		}

	def get_ds_cfg(self, model):
		cfg = {
			"train_micro_batch_size_per_gpu": self.hyperparameters.batch_size,
			"gradient_accumulation_steps": self.hyperparameters.gradient_accumulation_steps,
			"optimizer": {
				"type": self.hyperparameters.optimizer,
				"params": {
					"lr": self.hyperparameters.learning_rate,
				}
			},
			"scheduler": self.hyperparameters.scheduler if self.hyperparameters.scheduler_type != "" else None,
			"gradient_clipping": self.hyperparameters.gradient_clipping,
			"fp16": self.fp16_cfg,
			"bf16": self.bf16_cfg,
			"compression_training": self.get_compression_cfg(model),
			"zero_optimization": self.zero_cfg,
			"comms_logger": {
				"enabled": False
			}
		}

		null_keys = [ k for k in cfg if not cfg[k] ]
		for k in null_keys:
			del cfg[k]

		if os.path.exists("./config/ds_config.json"):
			ds_cfg = json.load(open("./config/ds_config.json", "r", encoding="utf-8"))
			cfg.update(ds_cfg)

		return cfg

	@property
	def cache_dir(self):
		return ".cache" / self.relpath

	@cached_property
	def diskcache(self):
		if self.dataset.cache:
			return diskcache.Cache(self.cache_dir).memoize
		return lambda: lambda x: x

	def load_yaml( self, config_path ):
		tmp = Config.from_yaml( config_path )
		self.__dict__.update(tmp.__dict__)


cfg = Config.from_cli()

# OmegaConf doesn't actually coerce the dicts into the @dataclass decorated classes, for some god forsaken reason, so we coerce them ourselves
cfg.dataset = Dataset(**cfg.dataset)
cfg.models = Models(**cfg.models)
cfg.hyperparameters = Hyperparameters(**cfg.hyperparameters)
cfg.evaluation = Evaluation(**cfg.evaluation)
cfg.trainer = Trainer(**cfg.trainer)
cfg.inference = Inference(**cfg.inference)
cfg.bitsandbytes = BitsAndBytes(**cfg.bitsandbytes)

# cached_property stopped working...
if cfg.dataset.use_hdf5:
	try:
		cfg.hdf5 = h5py.File(f'{cfg.cfg_path}/{cfg.dataset.hdf5_name}', 'a')
	except Exception as e:
		print("Error while opening HDF5 file:", f'{cfg.cfg_path}/{cfg.dataset.hdf5_name}', str(e))

if __name__ == "__main__":
	print(cfg)