"""
# https://github.com/enhuiz/pytorch-training-utilities
"""

import humanize
import json
import logging
import numpy as np
import random
import selectors
import sys
import torch
import os

from functools import cache
from torch.distributed import broadcast_object_list
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Protocol

from ..config import cfg
from .distributed import init_distributed, distributed_initialized, world_size
from .distributed import (
	global_leader_only,
	global_rank,
	is_global_leader,
	is_local_leader,
	local_leader_only,
)

from ..engines import _Engine, Engine, Engines, TrainFeeder, default_feeder
from ..models import get_models

from .utils import to_device, do_gc
from ..utils import wrapper as ml
from ..data import get_phone_symmap # should decouple from this trainer script

_logger = logging.getLogger(__name__)
_command: str

def load_engines(invert=False):
	models = get_models(cfg.models.get())
	engines = dict()

	for name, model in models.items():
		if cfg.mode != "inferencing":
			# load only the models for training initially
			# loads disabled models at evaluation time (to load updated weights if training separately)
			# I'm sure there's a more elegant solution to this
			if cfg.evaluation.load_disabled_engines:
				if not invert and not model._cfg.training:
					continue
				if invert and model._cfg.training:
					continue
			# load only the models for training initially
			# if load_disabled_engines, then models not marked for training will be loaded but ignored
			# DeepSpeed has some weird quirks where loading an engine and moving it to CPU will have a memory leak or something
			# I recommend not using this pathway
			elif not cfg.trainer.load_disabled_engines:
				if model._cfg.training:
					continue

		optimizer = None
		lr_scheduler = None

		if cfg.trainer.backend == "local" or (cfg.trainer.backend == "deepspeed" and cfg.hyperparameters.torch_optimizer):
			if cfg.hyperparameters.optimizer.lower() == "adamw":
				params = {
					"lr": cfg.hyperparameters.learning_rate,
					"betas": (0.9, 0.96),
					"eps": 1e-07,
					"weight_decay": 0.01,
				}
				params.update(cfg.hyperparameters.optimizer_params)
				optimizer = ml.AdamW(
					[ param for name, param in model.named_parameters() if name not in model._cfg.frozen_params ],
					**params,
				)
			elif cfg.hyperparameters.optimizer.lower() == "sgd":
				params = {
					"lr": cfg.hyperparameters.learning_rate,
				}
				params.update(cfg.hyperparameters.optimizer_params)
				optimizer = ml.SGD(
					[ param for name, param in model.named_parameters() if name not in model._cfg.frozen_params ],
					**params,
				)
			elif cfg.hyperparameters.optimizer.lower() == "prodigy":
				params = {
					"lr": cfg.hyperparameters.learning_rate,
				}
				params.update(cfg.hyperparameters.optimizer_params)
				optimizer = ml.Prodigy(
					[ param for name, param in model.named_parameters() if name not in model._cfg.frozen_params ],
					**params,
				)

		if not model._cfg.training:
			optimizer = None
			lr_scheduler = None

		if cfg.trainer.load_state_dict or not model._cfg.training:
			load_path = cfg.ckpt_dir / name / "fp32.pth"
			state = torch.load(load_path)
			# exporting the model from the zero_to_fp32.py exports the actual module's dict
			# exporting with vall_e.export exports the state dict under .module
			if "module" in state:
				state = state["module"]
			
			# should decouple the following from this trainer script
			# probably with passing a fun that defaults to a lambda x: x deal

			"""
			# can probably be done a lot more intelligently but oh well
			# extend the proms_emb if we ever touch the n_prom_levels or n_prom_tokens (from adding tasks)
			if model.proms_emb.weight.shape[0] > state['proms_emb.weight'].shape[0] or model.proms_emb.weight.shape[1] > state['proms_emb.weight'].shape[1]:
				o_prom_levels, o_prom_tokens, d_model = state['proms_emb.weight'].shape

				# copy weights from the dict into the old portion
				model.proms_emb.weight.data[:o_prom_levels, :o_prom_tokens, :] = state['proms_emb.weight'].data[:o_prom_levels, :o_prom_tokens, :]
				# copy the full tensors back
				state['proms_emb.weight'] = model.proms_emb.weight

			# extend the resps_emb if we ever touch the n_prom_levels or n_prom_tokens (from adding tasks)
			if model.resps_emb.weight.shape[0] > state['resps_emb.weight'].shape[0] or model.resps_emb.weight.shape[1] > state['resps_emb.weight'].shape[1]:
				o_resp_levels, o_resp_tokens, d_model = state['resps_emb.weight'].shape
				n_resp_levels, n_resp_tokens, d_model = model.resps_emb.weight.shape

				# copy weights from the dict into the old portion
				model.resps_emb.weight.data[:o_resp_levels, :o_resp_tokens, :] = state['resps_emb.weight'].data[:o_resp_levels, :o_resp_tokens, :]
				# copy the full tensors back
				state['resps_emb.weight'] = model.resps_emb.weight
			"""

			model.load_state_dict(state, strict=cfg.trainer.strict_loading)

		# use base engine because DeepSpeed memory leaks
		engines[name] = (Engine if model._cfg.training else _Engine)(
		#engines[name] = Engine(
			model=model,
			optimizer=optimizer,
			lr_scheduler=lr_scheduler,

			_cfg=model._cfg,
		)

	engines = Engines(engines)
	engines.setup()

	if not cfg.trainer.load_state_dict:
		engines.load_checkpoint()

	# freeze requested params
	for name, engine in engines.items():
		engine.freeze(freeze_all=False)

	do_gc()

	return engines

class EvalFn(Protocol):
	def __call__(self, *, engines: Engines):
		...


class Logger(Protocol):
	def __call__(self, *, data: dict):
		...


@cache
def _get_stdin_selector():
	selector = selectors.DefaultSelector()
	selector.register(fileobj=sys.stdin, events=selectors.EVENT_READ)
	return selector


if os.name == "nt":
	import msvcrt
	_buffer = []

def _non_blocking_input():
	global _command
	global _buffer
	l = [""]

	def _windows():
		global _buffer

		if msvcrt.kbhit():
			s: str = msvcrt.getch().decode('utf-8')
			if s == '\r':
				s = "".join(_buffer)
				_buffer = []
				return s

			_buffer.append(s)
		return ""

	def _linux():
		s = ""
		selector = _get_stdin_selector()
		events = selector.select(timeout=0)
		for key, _ in events:
			s: str = key.fileobj.readline().strip()
		return s

	if is_global_leader():
		s = _windows() if os.name == 'nt' else _linux()
	
		if s != "":
			_logger.info(f'Get stdin "{s}".')

		l[0] = s

	if world_size() > 1:
		broadcast_object_list(l, src=0)
	_command = l[0]
	return _command


def _make_infinite_epochs(dl):
	while True:
		_logger.info("New epoch starts.")
		yield from tqdm(dl, "Epoch progress", dynamic_ncols=True)


@local_leader_only(default=None)
def logger(data):
	return _logger.info(json.dumps(data, default=str))


def seed(seed):
	# Set up random seeds, after fork()
	random.seed(seed + global_rank())
	np.random.seed(seed + global_rank())
	torch.manual_seed(seed + global_rank())


def train(
	train_dl: DataLoader,
	train_feeder: TrainFeeder = default_feeder,
	eval_fn: EvalFn = lambda x: ...,
	logger: Logger = logger,
):
	engines = load_engines()

	"""
	if is_local_leader():
		cfg.dump()
		_logger.info(cfg)
	"""

	# Setup global engines
	global _engines
	_engines = engines

	events = []

	eval_fn = global_leader_only(eval_fn)

	# Pre-loop command
	command = _non_blocking_input()
	if command in ["eval", "eval_quit"]:
		engines.eval()
		eval_fn(engines=engines)
		engines.train()
	if command in ["quit", "eval_quit"]:
		return

	last_save_step = engines.global_step
	last_eval_step = 0

	# Training loop
	for batch in _make_infinite_epochs(train_dl):
		if engines.global_step >= cfg.trainer.iterations:
			break

		#batch = to_device(batch, torch.cuda.current_device())
		stats = engines.step(batch=batch, feeder=train_feeder)

		stats['it'] = stats['global_step']
		stats['epoch'] = engines.global_samples / len(train_dl.dataset.paths)

		stats['batch'] = {
			'size': len(batch['text']),
			'id': batch['spkr_id'],
			'index': [ index for index in batch['index'] ],
			'text_len': [ text.shape[0] for text in batch['text'] ],
			'prom_len': [ prom.shape[0] for prom in batch['proms'] ],
			'resp_len': [ resp.shape[0] for resp in batch['resps'] ],
		}

		del stats['global_step']

		elapsed_time = stats.get("elapsed_time", 0)
		_logger.info(f"Training Metrics: {json.dumps(stats)}.")

		command = _non_blocking_input()

		if "@" in command:
			what, when = command.split("@")
			try:
				events.append((what, int(when)))
				_logger.info(f"Event {command} registered.")
			except Exception as e:
				_logger.error(e)
			command = ""

		# Commands are the current command plus the triggered (i.e. iteration >= trigger point) events
		events = [e for e in events if e[1] >= engines.global_step]
		commands = [command] + [e[0] for e in events if e[1] == engines.global_step]

		for command in commands:
			if command in ["event show", "event"]:
				msg = "Events:\n" + "\n".join(["@".join(map(str, e)) for e in events])
				_logger.info(msg)

			if command == "event clear":
				events.clear()

			if "time" in command:
				target_iter = cfg.trainer.iterations
				if " to " in command:
					try:
						target_iter = int(command.split(" to ")[-1])
					except Exception as e:
						_logger.error(e)
				remaining_iters = target_iter - engines.global_step + 1
				remaining_time = int(remaining_iters * elapsed_time)
				_logger.info(humanize.precisedelta(remaining_time))

			if "lr" in command:
				rate = float(command.split(" ")[-1])
				try:
					engines.set_lr(rate)
					print("Updating LR to:", rate)
				except Exception as e:
					print("Failed to set LR rate to:", rate, str(e))

			if "export" in command:
				train_dl.dataset.save_state_dict(cfg.relpath / "train_dataset.pt")
				engines.save_checkpoint()
				last_save_step = engines.global_step

				if is_global_leader():
					engines.export(userdata={"symmap": get_phone_symmap()})

			save_ckpt_every = cfg.trainer.save_frequency or cfg.evaluation.frequency

			saving_commands = ["save"]
			export_commands = ["export"]

			if cfg.trainer.save_on_quit:
				saving_commands.append("quit")

			if cfg.trainer.export_on_quit:
				export_commands.append("quit")

			if cfg.trainer.export_on_save:
				export_commands.append("save")

			if engines.global_step != last_save_step:
				if engines.global_step % save_ckpt_every == 0 or command in saving_commands:
					train_dl.dataset.save_state_dict(cfg.relpath / "train_dataset.pt")
					engines.save_checkpoint()
					last_save_step = engines.global_step
					
					if command in export_commands and is_global_leader():
						engines.export(userdata={"symmap": get_phone_symmap()})

			if engines.global_step != last_eval_step:
				if engines.global_step % cfg.evaluation.frequency == 0 or command in ["eval"]:
					do_gc()

					engines.eval()
					eval_fn(engines=engines)
					engines.train()
					last_eval_step = engines.global_step

			if command in ["quit"]:
				return
