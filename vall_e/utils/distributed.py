"""
# https://github.com/enhuiz/pytorch-training-utilities
"""

import os
import socket

from functools import cache, wraps
from typing import Callable

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def get_free_port():
	sock = socket.socket()
	sock.bind(("", 0))
	return sock.getsockname()[1]


_distributed_initialized = False
def init_distributed( fn, *args, **kwargs ):
	torch.cuda.set_device(local_rank())
	fn(*args, **kwargs)
	_distributed_initialized = True

def distributed_initialized():
	return _distributed_initialized

def cleanup_distributed():
	dist.barrier()
	dist.destroy_process_group()

@cache
def fix_unset_envs():
	envs = dict(
		RANK="0",
		WORLD_SIZE="1",
		MASTER_ADDR="localhost",
		MASTER_PORT=str(get_free_port()),
		LOCAL_RANK="0",
	)

	for key in envs:
		value = os.getenv(key)
		if value is not None:
			return

	for key, value in envs.items():
		os.environ[key] = value


def local_rank():
	return int(os.getenv("LOCAL_RANK", 0))

def global_rank():
	return int(os.getenv("RANK", 0))

def world_size():
	return int(os.getenv("WORLD_SIZE", 1))


def is_local_leader():
	return local_rank() == 0


def is_global_leader():
	return global_rank() == 0


def local_leader_only(fn=None, *, default=None) -> Callable:
	def wrapper(fn):
		@wraps(fn)
		def wrapped(*args, **kwargs):
			if is_local_leader():
				return fn(*args, **kwargs)
			return default

		return wrapped

	if fn is None:
		return wrapper

	return wrapper(fn)


def global_leader_only(fn: Callable | None = None, *, default=None) -> Callable:
	def wrapper(fn):
		@wraps(fn)
		def wrapped(*args, **kwargs):
			if is_global_leader():
				return fn(*args, **kwargs)
			return default

		return wrapped

	if fn is None:
		return wrapper

	return wrapper(fn)

def ddp_model(model):
	return DDP(model.to(device='cuda'), [local_rank()], find_unused_parameters=True)