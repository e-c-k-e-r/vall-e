from ..config import cfg

from ..utils.distributed import fix_unset_envs
fix_unset_envs()

if cfg.trainer.backend == "deepspeed":
	from .deepspeed import Engine
elif cfg.trainer.backend == "local":
	from .base import Engine

from .base import Engines, TrainFeeder, default_feeder, Engine as _Engine

from ..models import get_models
from ..utils import wrapper as ml
import torch
import re

deepspeed_available = False
try:
	import deepspeed
	deepspeed_available = True
except Exception as e:
	pass

from functools import cache

@cache
def load_engines(training=True):
	models = get_models(cfg.models.get(), training=training)
	engines = dict()

	for name, model in models.items():
		optimizer = None
		lr_scheduler = None

		inferencing = cfg.mode == "inferencing" or not model._cfg.training
		backend = cfg.inference.backend if inferencing else cfg.trainer.backend
		dtype = cfg.inference.dtype if inferencing else cfg.trainer.dtype
		amp = cfg.inference.amp if inferencing else cfg.trainer.amp
		loads_state_dict = cfg.trainer.load_state_dict or inferencing

		engine_class = _Engine if backend == "local" or inferencing else Engine

		if inferencing:
			model._cfg.training = False

		if cfg.bitsandbytes.enabled and cfg.bitsandbytes.replace:
			model.model = ml.replace_linear( model.model )

		if backend == "local" or (backend == "deepspeed" and cfg.hyperparameters.torch_optimizer):
			optimizer_class = None
			params = {
				"lr": cfg.hyperparameters.learning_rate,
			}
			if cfg.hyperparameters.optimizer.lower() == "adamw":
				params["betas"] = (0.9, 0.96)
				params["eps"] = 1e-07
				params["weight_decay"] = 0.01

				optimizer_class = ml.AdamW
			elif cfg.hyperparameters.optimizer.lower() == "sgd":
				optimizer = ml.SGD
			elif cfg.hyperparameters.optimizer.lower() == "prodigy":
				optimizer_class = ml.Prodigy

				params['d_coef'] = params['lr']
				params['lr'] = 1.0
			else:
				raise ValueError(f'Optimizer specified not implemented: {cfg.hyperparameters.optimizer}')

			params.update(cfg.hyperparameters.optimizer_params)
			optimizer = optimizer_class(
				[ param for name, param in model.named_parameters() if name not in model._cfg.frozen_params ],
				**params,
			)

		# set up our LR scheduler here

		if inferencing:
			optimizer = None
			lr_scheduler = None

		# automatically load from state dict if one is provided, but no DeepSpeed checkpoint is present
		load_path = cfg.ckpt_dir / name / "fp32.pth"
		if not loads_state_dict and backend == "deepspeed" and not (cfg.ckpt_dir / name / "latest").exists() and load_path.exists():
			print("DeepSpeed checkpoint missing, but weights found.")
			loads_state_dict = True

		stats = None
		if loads_state_dict:
			state = torch.load(load_path, map_location=torch.device(cfg.device))

			# state dict is not just the module, extract the extra trainer details
			if "stats" in state:
				stats = state["stats"]

			if "module" in state:
				state = state["module"]

			# maintain compat if I change variable names
			insert = {}
			erase = []

			for k in state.keys():
				key = re.sub(r'^retnet\.', "model.", k)
				if k != key:
					insert[key] = state[k]
					erase.append(k)
	
			for k in insert.keys():
				state[k] = insert[k]

			for k in erase:
				del state[k]

			model.load_state_dict(state, strict=cfg.trainer.strict_loading)

		# deepspeed inferencing
		if backend == "local" and inferencing and deepspeed_available and cfg.trainer.deepspeed.inferencing: #and sys.platform.startswith("win"):
			engine_class = _Engine
			model = deepspeed.init_inference(model=model, mp_size=1, replace_with_kernel_inject=True, dtype=dtype if not amp else torch.float32).module

		# use base engine if requested
		engines[name] = engine_class(
			model=model,
			optimizer=optimizer,
			lr_scheduler=lr_scheduler,

			_cfg=model._cfg,
			stats=stats
		)

	engines = Engines(engines)
	engines.setup()

	if not cfg.trainer.load_state_dict:
		engines.load_checkpoint()

	# freeze requested params
	for name, engine in engines.items():
		engine.freeze(freeze_all=False)

	#do_gc()

	return engines