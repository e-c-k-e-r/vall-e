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

deepspeed_available = False
try:
	import deepspeed
	deepspeed_available = True
except Exception as e:
	pass

def load_engines():
	models = get_models(cfg.models.get())
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
		if not loads_state_dict and backend == "deepspeed" and not (cfg.ckpt_dir / name / "latest").exists():
			print("DeepSpeed checkpoint missing, but weights found.")
			loads_state_dict = True

		stats = None
		if loads_state_dict:
			load_path = cfg.ckpt_dir / name / "fp32.pth"
			state = torch.load(load_path, map_location=torch.device(cfg.device))

			# state dict is not just the module, extract the extra trainer details
			if "stats" in state:
				stats = state["stats"]

			if "module" in state:
				state = state["module"]

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