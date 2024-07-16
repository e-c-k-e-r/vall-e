from ..config import cfg

from ..utils.distributed import fix_unset_envs, ddp_model
fix_unset_envs()

if cfg.trainer.backend == "deepspeed":
	from .deepspeed import Engine
elif cfg.trainer.backend == "local":
	from .base import Engine

from .base import Engines, TrainFeeder, default_feeder, Engine as LocalEngine

from ..models import get_models, get_model
from ..utils import wrapper as ml
from ..models.lora import apply_lora

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
	models = get_models(cfg.models, training=training)
	engines = dict()

	for name, model in models.items():
		state = None
		stats = None
		lora = None

		inferencing = cfg.mode == "inferencing" or not model.config.training or not training
		loads_state_dict = cfg.trainer.load_state_dict or inferencing

		checkpoint_path = cfg.ckpt_dir / name / "latest"
		# automatically load from state dict if one is provided, but no DeepSpeed checkpoint is present
		load_path = cfg.ckpt_dir / name / "fp32.pth"

		# actually use the lora-specific checkpoint if available
		if cfg.lora is not None:
			lora = cfg.lora
			checkpoint_path = cfg.ckpt_dir / lora.full_name / "latest"

		if not loads_state_dict and not checkpoint_path.exists() and load_path.exists():
			print("Checkpoint missing, but weights found.")
			loads_state_dict = True

		# load state early
		if loads_state_dict:
			state = torch.load(load_path, map_location=torch.device(cfg.device))

			# check if config is defined in state, and re-initialize the model
			if "config" in state:
				print("Model config definition in weights, re-loading...")
				config_state = state["config"]
				model = get_model( config=cfg.model.__class__( *config_state ), training=training )

		hyper_config = model.config

		optimizer = None
		lr_scheduler = None

		backend = cfg.inference.backend if inferencing else cfg.trainer.backend
		dtype = cfg.inference.dtype if inferencing else cfg.trainer.dtype
		amp = cfg.inference.amp if inferencing else cfg.trainer.amp
		ddp = cfg.trainer.ddp

		engine_class = LocalEngine if backend == "local" or inferencing else Engine

		# apply model replacers
		if cfg.optimizations.replace and cfg.optimizations.linear:
			model.model = ml.replace_linear( model.model )
		
		if cfg.optimizations.replace and cfg.optimizations.embedding:
			model.model = ml.replace_embedding( model.model )

		for lora in cfg.loras:
			model.model = apply_lora( model.model, rank = lora.rank, alpha = lora.alpha, policy = model.config.lora_policy, use_parametrize = lora.parametrize )

		if inferencing:
			model.config.training = False

		if not inferencing and (backend == "local" or (backend == "deepspeed" and cfg.hyperparameters.torch_optimizer)):
			optimizer_class = None
			scheduler_class = None

			params = {
				"lr": cfg.hyperparameters.learning_rate,
			}
			if cfg.hyperparameters.optimizer.lower() == "adamw":
				params["betas"] = (0.9, 0.96)
				params["eps"] = 1e-07
				params["weight_decay"] = 0.01

				# for dadaptation since it has Adam only
				if ml.AdamW == ml.Adam:
					params["decouple"] = True

				optimizer_class = ml.AdamW
			elif cfg.hyperparameters.optimizer.lower() == "sgd":
				optimizer = ml.SGD
			elif cfg.hyperparameters.optimizer.lower() == "prodigy":
				optimizer_class = ml.Prodigy

				params['d_coef'] = params['lr']
				params['lr'] = 1.0
			elif cfg.hyperparameters.optimizer.lower() == "adagrad":
				optimizer_class = ml.Adagrad
			else:
				raise ValueError(f'Optimizer specified not implemented: {cfg.hyperparameters.optimizer}')

			params.update(cfg.hyperparameters.optimizer_params)

			optimizer = optimizer_class(
				[ param for name, param in model.named_parameters() if name not in model.config.frozen_params ],
				**params,
			)

			if cfg.hyperparameters.scheduler.lower() == "schedulefree":
				if cfg.hyperparameters.optimizer.lower() == "adamw":
					scheduler_class = ml.schedulefree.AdamWScheduleFree
				elif cfg.hyperparameters.optimizer.lower() == "sgd":
					scheduler_class = ml.schedulefree.SGDScheduleFree
				else:
					raise ValueError(f'ScheduleFree not implemented with requested optimizer: {cfg.hyperparameters.optimizer}')

				optimizer = scheduler_class(
					[ param for name, param in model.named_parameters() if name not in model.config.frozen_params ],
					lr = params['lr'],
					warmup_steps = cfg.hyperparameters.warmup_steps
				)

		"""
		# set up our LR scheduler here
		"""

		if inferencing:
			optimizer = None
			lr_scheduler = None

		# load state dict if requested / required
		if loads_state_dict:
			# state dict is not just the module, extract the extra trainer details
			if "stats" in state:
				stats = state["stats"]

			# do not load stats if we're training a LoRA
			if "lora" in state or cfg.lora is not None or cfg.trainer.restart_step_count:
				stats = None

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

			# resize embeddings
			if "text_emb.weight" in state:
				state["text_emb.weight"] = ml.resize_weight( state["text_emb.weight"], model.config.text_tokens )
			if "rvq_l_emb.weight" in state:
				state["rvq_l_emb.weight"] = ml.resize_weight( state["rvq_l_emb.weight"], model.config.resp_levels )

			model.load_state_dict(state, strict=cfg.trainer.strict_loading)

		# load lora weights if exists
		if cfg.lora is not None:
			lora_path = cfg.ckpt_dir / lora.full_name / "lora.pth"
			if lora_path.exists():
				state = torch.load(lora_path, map_location=torch.device(cfg.device))
				state = state['lora' if 'lora' in state else 'module']
				model.load_state_dict(state, strict=False)

		# wrap if DDP is requested
		if ddp:
			model = ddp_model(model)

		# deepspeed inferencing
		elif backend == "local" and inferencing and deepspeed_available and cfg.trainer.deepspeed.inferencing: #and sys.platform.startswith("win"):
			engine_class = LocalEngine
			model = deepspeed.init_inference(model=model, mp_size=1, replace_with_kernel_inject=True, dtype=dtype if not amp else torch.float32).module

		# use base engine if requested
		engines[name] = engine_class(
			model=model,
			optimizer=optimizer,
			lr_scheduler=lr_scheduler,

			hyper_config=hyper_config,
			stats=stats
		)
		

	engines = Engines(engines)
	engines.setup()

	if not cfg.trainer.load_state_dict:
		engines.load_checkpoint()

	# freeze requested params
	for name, engine in engines.items():
		engine.freeze(freeze_all=False)

		"""
		# copy embeddings if requested
		if cfg.model._embeddings is not None:
			embeddings_path = cfg.rel_path / cfg.model._embeddings
			
			if embeddings_path.exists():
				embeddings = torch.load(embeddings_path, map_location=torch.device(cfg.device))
				if "module" in embeddings:
					embeddings = embeddings["module"]

				frozen_params = set()

				for k in list(embeddings.keys()):
					if re.findall(r'_emb\.', k):
						frozen_params.add(k)
					else:
						del embeddings[k]

				engine.module.load_state_dict(embeddings, strict=False)

				# there's definitely a much better way but I can't be assed at the moment
				for name, param in engine.module.named_parameters():
					if name not in frozen_params:
						continue
					param.requires_grad_(False)
					engine._frozen_params.add(param)
		"""
			
		
	#do_gc()

	return engines