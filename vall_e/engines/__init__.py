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
from ..utils.io import torch_save, torch_load, pick_path
from ..models.lora import apply_lora, lora_load_state_dict

import torch
import re
import logging

_logger = logging.getLogger(__name__)

deepspeed_available = False
try:
	import deepspeed
	deepspeed_available = True
except Exception as e:
	pass

from functools import cache

@cache
def load_engines(training=True, **model_kwargs):
	models = get_models(cfg.models, training=training, **model_kwargs)
	engines = dict()

	for name, model in models.items():
		state = None
		stats = None
		lora = None

		inferencing = cfg.mode == "inferencing" or not model.config.training or not training
		backend = cfg.inference.backend if inferencing else cfg.trainer.backend
		loads_state_dict = cfg.trainer.load_state_dict # or inferencing

		checkpoint_path = cfg.ckpt_dir / name / "latest"
		# automatically load from state dict if one is provided, but no DeepSpeed checkpoint is present
		load_path = pick_path( cfg.ckpt_dir / name / f"fp32.{cfg.weights_format}", *[ f'.{format}' for format in cfg.supported_weights_formats] )

		# actually use the lora-specific checkpoint if available
		if cfg.lora is not None:			
			checkpoint_path = cfg.ckpt_dir / cfg.lora.full_name / "latest"

		# to handle the issue of training with deepspeed, but inferencing with local
		if checkpoint_path.exists() and backend == "local":
			tag = open(checkpoint_path).read()
			checkpoint_path = pick_path( checkpoint_path.parent / tag / f"state.{cfg.weights_format}", *[ f'.{format}' for format in cfg.supported_weights_formats] )

		if not loads_state_dict and not checkpoint_path.exists() and load_path.exists():
			_logger.warning(f"Checkpoint missing, but weights found: {load_path}")
			loads_state_dict = True

		# load state early
		if loads_state_dict:
			state = torch_load(load_path, device=cfg.device)

			# check if config is defined in state, and re-initialize the model
			if "config" in state and False:
				_logger.warning("Model config definition in weights, re-loading...")
				config_state = state["config"]
				model = get_model( config=cfg.model.__class__( *config_state ), training=training )

		hyper_config = model.config

		optimizer = None
		lr_scheduler = None

		dtype = cfg.inference.dtype if inferencing else cfg.trainer.dtype
		amp = cfg.inference.amp if inferencing else cfg.trainer.amp
		ddp = cfg.trainer.ddp

		engine_class = LocalEngine if backend == "local" else Engine

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
			if cfg.lora is not None or cfg.trainer.restart_step_count:
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

			# resize modules if I'm doing experiments and can't be assed to manually trim things
			if cfg.trainer.resize_modules:
				uses_stop_token = 1 if "len" not in model.capabilities and model.causal_size > 0 else 0
				keys = [
					("text_emb.weight", model.config.text_tokens ),
					("tasks_emb.weight", model.config.tasks ),
					("langs_emb.weight", model.config.langs ),
					("rvq_l_emb.weight", model.config.resp_levels + (1 if "len" in model.config.capabilities else 0) ),
					("resps_emb.embeddings.0.weight", model.config.audio_tokens + uses_stop_token ),
					("model.embed_tokens.weight", model.config.audio_tokens + uses_stop_token ),
					("classifiers.proj.0.weight" if model.config.experimental.split_classifiers else 'classifier.weight', model.config.audio_tokens + uses_stop_token ),
					("classifiers.proj.0.bias" if model.config.experimental.split_classifiers else 'classifier.bias', model.config.audio_tokens + uses_stop_token ),
				]
				for k, tokens in keys:
					if k not in state:
						continue
					state[k] = ml.resize_weight( state[k], tokens )

			model.load_state_dict(state, strict=cfg.trainer.strict_loading)

			# load lora weights if exists
			if cfg.lora is not None:
				lora_path = pick_path( cfg.ckpt_dir / cfg.lora.full_name / f"lora.{cfg.weights_format}", *[ f'.{format}' for format in cfg.supported_weights_formats] )
				if lora_path.exists():
					_logger.info( f"Loaded LoRA state dict: {lora_path}" )

					state = torch_load(lora_path, device=cfg.device)
					state = state['lora' if 'lora' in state else 'module']
					lora_load_state_dict( model, state )

		# wrap if DDP is requested
		if ddp:
			model = ddp_model(model)
		# wrap optimization class
		elif cfg.optimizations.compile:
			model = ml.compile_model(model, backend=cfg.optimizations.compile)
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

	# this might bite me in the ass since technically this doesn't handle one engine loading fine but another engine not
	if not cfg.trainer.load_state_dict:
		engines.load_checkpoint(training=not inferencing)

	# freeze requested params
	for name, engine in engines.items():
		engine.freeze(freeze_all=False)

		# split models over requested devices
		if cfg.optimizations.model_offloading:
			engine.module = ml.offload_model( engine.module, policy=cfg.optimizations.model_offloading )

	return engines
