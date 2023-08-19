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
from .distributed import init_distributed, distributed_initialized
from .distributed import (
	global_leader_only,
	global_rank,
	is_global_leader,
	is_local_leader,
	local_leader_only,
)

from ..engines import Engine, Engines, TrainFeeder, default_feeder
from ..models import get_models

from .utils import to_device, do_gc
from ..utils import wrapper as ml

_logger = logging.getLogger(__name__)
_engines: Engines
_command: str

def get_global_step():
	try:
		return _engines.global_step
	except:
		return None

def get_micro_step():
	try:
		return _engines.micro_step
	except:
		return None

def get_cmd():
	try:
		return _command
	except:
		raise RuntimeError("Trainer has not been setup. Have you called trainer.train?")


get_iteration = get_global_step

def load_engines():
	models = get_models(cfg.models.get())
	engines = dict()

	for name in models:
		model = models[name]

		optimizer = None
		lr_scheduler = None

		if cfg.hyperparameters.optimizer.lower() == "adamw-torch":
			optimizer = ml.AdamW(
				model.parameters(),
				lr=cfg.hyperparameters.learning_rate,
				betas=(0.9, 0.96),
				eps=1e-07,
				weight_decay=0.01,
			)

		if cfg.trainer.load_state_dict:
			load_path = cfg.ckpt_dir / name / "fp32.pth"
			state = torch.load(load_path)
			# exporting the model from the zero_to_fp32.py exports the actual module's dict
			# exporting with vall_e.export exports the state dict under .module
			if "module" in state:
				state = state["module"]
			
			print(model.proms_emb.weight.shape, state['proms_emb.weight'].shape)

			# extend the proms_emb if we ever touch the n_prom_levels or n_prom_tokens (from adding tasks)
			if model.proms_emb.weight.shape[0] > state['proms_emb.weight'].shape[0] or model.proms_emb.weight.shape[1] > state['proms_emb.weight'].shape[1]:
				n_prom_levels, n_prom_tokens, d_model = state['proms_emb.weight'].shape

				model.proms_emb.weight.data[:n_prom_levels, :n_prom_tokens, :] = state['proms_emb.weight'].data[:n_prom_levels, :n_prom_tokens, :]
				state['proms_emb.weight'] = model.proms_emb.weight

			model.load_state_dict(state, strict=cfg.trainer.strict_loading)

		engines[name] = Engine(
			model=model,
			optimizer=optimizer,
			lr_scheduler=lr_scheduler,
		)

	engines = Engines(engines)
	engines.setup()

	if not cfg.trainer.load_state_dict:
		engines.load_checkpoint()

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

		iteration = stats['global_step'] # * cfg.hyperparameters.gradient_accumulation_steps
		stats['it'] = iteration
		stats['epoch'] = iteration * cfg.hyperparameters.gradient_accumulation_steps / len(train_dl)

		stats['batch'] = {
			'size': stats['batch_size'],
			'id': batch['spkr_id'],
			'index': [ index for index in batch['index'] ],
			'text_len': [ text.shape[0] for text in batch['text'] ],
			'prom_len': [ prom.shape[0] for prom in batch['proms'] ],
			'resp_len': [ resp.shape[0] for resp in batch['resps'] ],
		}

		del stats['batch_size']
		del stats['wall_time']
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
				engines.set_lr(rate)
				print("Updating LR to:", rate)

			save_ckpt_every = cfg.trainer.save_frequency or cfg.evaluation.frequency

			saving_commands = ["save"]

			if cfg.trainer.save_on_quit:
				saving_commands.append("quit")

			if engines.global_step != last_save_step:
				if engines.global_step % save_ckpt_every == 0 or command in saving_commands:
					engines.save_checkpoint()
					last_save_step = engines.global_step

			if engines.global_step != last_eval_step:
				if engines.global_step % cfg.evaluation.frequency == 0 or command in ["eval"]:
					do_gc()

					engines.eval()
					eval_fn(engines=engines)
					engines.train()
					last_eval_step = engines.global_step

			if command in ["quit"]:
				return
