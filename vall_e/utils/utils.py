"""
# https://github.com/enhuiz/pytorch-training-utilities
"""

from .distributed import global_rank, local_rank, global_leader_only

import gc
import logging
import pandas as pd
import numpy as np
import re
import torch
import random
import time
import psutil
import math
import logging
import hashlib

_logger = logging.getLogger(__name__)

from coloredlogs import ColoredFormatter
from logging import StreamHandler
from pathlib import Path
from torch import Tensor, nn
from tqdm.auto import tqdm
from typing import Callable, TypeVar, overload
from contextlib import contextmanager

from time import perf_counter
from datetime import datetime

T = TypeVar("T")

def mean( l ):
	if not l:
		return 0
	return sum(l) / len(l)

def logit_normalization( logit, factor=1, eps=1.0e-7 ):
	norms = torch.norm(logit, p=2, dim=-1, keepdim=True) + eps
	return torch.div(logit, norms) / factor

# removes prefix from key in a dict
# useful for mapping args like ar_temperature => temperature
def convert_kwargs( kwargs, prefix ):
	copied = {} | kwargs

	for key, value in copied.items():
		if not key.startswith( prefix ):
			continue

		kwargs.pop(key)
		kwargs[key[len(prefix):]] = value

	return kwargs

# hashes values or a list of values
def md5_hash( x ):
	if isinstance( x, list ):
		return md5_hash(":".join([ md5_hash( _ ) for _ in x ]))
	return hashlib.md5(str(x).encode("utf-8")).hexdigest()

# removes entries from a dict if that key is missing from the source
def prune_missing( source, dest, recurse=True, path=[], parent_is_obj=None, return_missing=True, ignore=["optimizer_params", "wandb_params"] ):
	is_obj = hasattr( source, "__dict__" )
	if parent_is_obj is None:
		parent_is_obj = is_obj
	haystack = source.__dict__ if is_obj else source
	keep = {}
	missing = []
	for k, v in dest.items():
		if k in haystack or (parent_is_obj and not is_obj and source == {}):
			keep[k] = dest[k]
		else:
			missing.append(".".join(path + [k]))

		if k in ignore:
			continue
		
		if recurse and isinstance( v, dict ):
			keep[k], m = prune_missing( haystack[k], dest[k], path=path + [k], parent_is_obj=parent_is_obj, return_missing=return_missing )
			missing += m
	return (keep, missing) if return_missing else keep

def clamp(n, lo, hi):
	return max(lo, min(n, hi))

class timer:
	def __init__(self, msg="Elapsed time:", callback=None):
		self.msg = msg
		self.callback = callback

	def __enter__(self):
		self.start = perf_counter()
		return self

	def __exit__(self, type, value, traceback):
		msg = f'{self.msg} {(perf_counter() - self.start):.9f}s'

		if self.callback:
			self.callback(msg)

		print(f'[{datetime.now().isoformat()}] {msg}')

def truncate_json( x ):
	if isinstance( x, bytes ):
		return truncate_json( x.decode('utf-8') ).encode()

	def fun( match ):
		return "{:.4f}".format(float(match.group()))

	return re.sub(r"\d+\.\d{8,}", fun, x)

def do_gc():
	gc.collect()
	torch.cuda.empty_cache()

def flatten_dict(d):
	records = pd.json_normalize(d).to_dict(orient="records")
	return records[0] if records else {}


def set_seed(seed=None):
	if not seed:
		seed = int(time.time())

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	return seed

def _get_named_modules(module, attrname):
	for name, module in module.named_modules():
		if hasattr(module, attrname):
			yield name, module

def coerce_dtype(s):
	# not a string
	if not isinstance(s, str):
		return s

	if s == "float16":
		return torch.float16
	if s == "bfloat16":
		return torch.bfloat16
	if s == "float8_e5m2":
		return torch.float8_e5m2
	if s == "float8_e4m3fn":
		return torch.float8_e4m3fn
	return torch.float32

def gather_attribute(module, attrname, delete=True, prefix=True):
	ret = {}
	for name, module in _get_named_modules(module, attrname):
		ret[name] = getattr(module, attrname)
		if delete:
			try:
				delattr(module, attrname)
			except Exception as e:
				raise RuntimeError(f"{name} {module} {attrname}") from e
	if prefix:
		ret = {attrname: ret}
	ret = flatten_dict(ret)
	# remove consecutive dots
	ret = {re.sub(r"\.+", ".", k): v for k, v in ret.items()}
	return ret


def dispatch_attribute(
	module,
	attrname,
	value,
	filter_fn: Callable[[nn.Module], bool] | None = None,
):
	for _, module in _get_named_modules(module, attrname):
		if filter_fn is None or filter_fn(module):
			setattr(module, attrname, value)


def load_state_dict_non_strict(model, state_dict, logger=None):
	model_state_dict = model.state_dict()
	provided = set(state_dict)
	required = set(model_state_dict)
	agreed = provided & required
	for k in list(agreed):
		if model_state_dict[k].shape != state_dict[k].shape:
			agreed.remove(k)
			provided.remove(k)
	state_dict = {k: state_dict[k] for k in agreed}
	if logger is not None and (diff := provided - required):
		logger.warning(
			f"Extra parameters are found. "
			f"Provided but not required parameters: \n{diff}."
		)
	if logger is not None and (diff := required - provided):
		logger.warning(
			f"Some parameters are missing. "
			f"Required but not provided parameters: \n{diff}."
		)
	model.load_state_dict(state_dict, strict=False)

class TqdmLoggingHandler(logging.Handler):
	def __init__(self, level=logging.INFO):
		super().__init__(level)

	def emit(self, record):
		try:
			msg = self.format(record)
			tqdm.write(msg)
			self.flush()
		except Exception as e:
			self.handleError(record) 

@global_leader_only
def setup_logging(log_dir: str | Path | None = None, log_level="info"):
	handlers = []

	#stdout_handler = StreamHandler()
	stdout_handler = TqdmLoggingHandler()
	stdout_handler.setLevel(logging.INFO)
	formatter = ColoredFormatter(
		f"%(asctime)s - %(name)s - %(levelname)s - GR={global_rank()};LR={local_rank()} - \n%(message)s"
	)
	stdout_handler.setFormatter(formatter)
	handlers.append(stdout_handler)

	if log_dir is not None:
		filename = Path(log_dir) / f"log.txt"
		filename.parent.mkdir(parents=True, exist_ok=True)
		file_handler = logging.FileHandler(filename, mode="a")
		file_handler.setLevel(logging.DEBUG)
		handlers.append(file_handler)


	logging.basicConfig(
		level=logging.getLevelName(log_level.upper()),
		format="%(asctime)s - %(name)s - %(levelname)s - \n%(message)s",
		handlers=handlers,
	)

@overload
def tree_map(fn: Callable, x: list[T]) -> list[T]:
	...


@overload
def tree_map(fn: Callable, x: tuple[T]) -> tuple[T]:
	...


@overload
def tree_map(fn: Callable, x: dict[str, T]) -> dict[str, T]:
	...


@overload
def tree_map(fn: Callable, x: T) -> T:
	...


def tree_map(fn: Callable, x):
	if isinstance(x, list):
		x = [tree_map(fn, xi) for xi in x]
	elif isinstance(x, tuple):
		x = (tree_map(fn, xi) for xi in x)
	elif isinstance(x, dict):
		x = {k: tree_map(fn, v) for k, v in x.items()}
	elif isinstance(x, Tensor):
		x = fn(x)
	return x


def to_device(x: T | None, *args, **kwargs) -> T:
	if x is None:
		return

	return tree_map(lambda t: t.to(*args, **kwargs), x)

def coalese( *arg, return_last=True ):
	return [ x for x in arg if x is not None ][-1 if return_last else 0]

# checks if a module name is within a given whitelist/blacklist policy dict
def passes_policy( policy, name ):
	if policy is None:
		return True

	if "exclude" in policy:
		for term in policy["exclude"]:
			if term in name:
				return False

	if "include" in policy:
		for term in policy["include"]:
			if term in name:
				return True

	return False

# handles generically converting to a specific tensor type and converting back (implemented solely for bfloat16)
@contextmanager
def autocast(input, from_dtype, to_dtype):
	if input.dtype == from_dtype:
		input = input.to(to_dtype)
		yield input
		input = input.to(from_dtype)
	else:
		yield input

@contextmanager
def autocasts(input, from_dtype, to_dtype):
	if input.dtype in from_dtype:
		from_dtype = input.dtype
		input = input.to(to_dtype)
		yield input
		input = input.to(from_dtype)
	else:
		yield input

# handles temporarily upcasting 'index tensors' so torch will stop bitching
def autocast_forward( func ):
	def wrapper( self, input, *args, **kwargs ):
		with autocasts( input, [torch.int16, torch.int8, torch.uint8, torch.float16, torch.bfloat16], torch.int32 ) as k:
			return func( self, k, *args, **kwargs )
	return wrapper

# handles migrating an input tensor to a given devicve
def auto_align_inputs_forward( module, device=None, name = None ):
	func = module.forward

	if device is None:
		if hasattr( module, 'device' ):
			device = module.device
		else:
			try:
				device = next(module.parameters() if [*module.parameters()] else module.buffers()).device
			except Exception as e:
				return func


	def wrapper( *args, **kwargs ):
		args = [*args]
		# search through args and kwargs for any Tensor arguments
		for i, arg in enumerate(args):
			if not isinstance( arg, torch.Tensor ):
				continue
			args[i] = arg.to( device=device )

		for k, v in kwargs.items():
			if not isinstance( v, torch.Tensor ):
				continue
			kwargs[k] = v.to( device=device )

		# disgusting patch
		if "position_embeddings" in kwargs:
			kwargs["position_embeddings"] = tuple([ t.to(device=device) for t in kwargs["position_embeddings"] ])

		return func( *args, **kwargs )
	return wrapper

# disgusting kludge, but it works (just realized BitNet has its own replacement routine)
# generalizing this would be super sugoi but the there's no catch all for arguments
def replace_linear( model, klass, target=torch.nn.Linear, verbose=False ):
	bnb = cfg.optimizations.bitsandbytes and cfg.optimizations.linear and not cfg.optimizations.bitnet

	device =  next(model.parameters()).device
	dtype = next(model.parameters()).dtype
	modules = [k.split('.') for k, m in model.named_modules() if isinstance(m, target)]

	for *parent, k in modules:
		name = '.'.join(parent)

		m = getattr( model.get_submodule(name), k )

		if isinstance(m, klass):
			continue

		kwargs = dict(
			in_features = m.in_features,
			out_features = m.out_features,
			bias = m.bias is not None,
		) if not bnb else dict(
			input_features=m.in_features,
			output_features=m.out_features,
			bias=m.bias is not None,
		)

		# overwrite
		setattr(
			model.get_submodule(name), k,
			klass( **kwargs ).to(device=device, dtype=dtype)
		)
		
		if verbose:
			_logger.info(f"Replacing {name}.{k} to: {klass}")

	return model

def replace_embedding( model, klass, target=torch.nn.Embedding, verbose=False ):
	device =  next(model.parameters()).device
	dtype = next(model.parameters()).dtype
	modules = [k.split('.') for k, m in model.named_modules() if isinstance(m, target)]

	for *parent, k in modules:
		name = '.'.join(parent)

		m = getattr( model.get_submodule(name), k )

		if isinstance(m, klass):
			continue

		kwargs = dict(
			num_embeddings=m.num_embeddings,
			embedding_dim=m.embedding_dim,
			padding_idx=m.padding_idx,
			max_norm=m.max_norm,
			norm_type=m.norm_type,
			scale_grad_by_freq=m.scale_grad_by_freq,
			sparse=m.sparse,
		)

		# overwrite
		setattr(
			model.get_submodule(name), k,
			klass( **kwargs ).to(device=device, dtype=dtype)
		)
		
		if verbose:
			_logger.info(f"Replacing {name}.{k} to: {klass}")

	return model

# cannot feasibly do default arguments here sad
def replace_attention( model, klass, target, mode="math", verbose=False ):
	device = next(model.parameters()).device
	dtype = next(model.parameters()).dtype
	modules = [k.split('.') for k, m in model.named_modules() if isinstance(m, target)]

	for *parent, k in modules:
		name = '.'.join(parent)

		m = getattr( model.get_submodule(name), k )

		if isinstance(m, klass):
			continue

		kwargs = dict(
			config = m.config,
			layer_idx = m.layer_idx,
			mode = mode,
		)
		# overwrite
		setattr(
			model.get_submodule(name), k,
			klass( **kwargs ).to(device=device, dtype=dtype)
		)
		
		if verbose:
			_logger.info(f"Replacing {name}.{k} to: {klass}")

	return model

# trim/expand a tensor (for example, in a state dict)
def resize_weight( weight, target, dim=0, random=True ):
	# trim
	if target < weight.shape[dim]:
		return weight[:target]
	# expand
	if target > weight.shape[dim]:
		fn = torch.rand if random else torch.zeros
		return torch.stack(
			[ x for x in weight ] +
			[ fn( weight[0].shape ).to(device=weight[0].device, dtype=weight[0].dtype) for _ in range( target - weight.shape[dim] ) ]
		)

	return weight

def get_devices():
	return [f'{"cuda"}:{i}' for i in range(torch.cuda.device_count())] + ['cpu']

# grabs the memory properties of a given device
def get_device_properties( device ):
	if 'cuda' in device:
		props = torch.cuda.get_device_properties(device)
		free, total = torch.cuda.mem_get_info(device)
	else:
		props = psutil.virtual_memory()
		free, total = props.available, props.total

	return {"name": device, "total": total, "free": free, "props": props}

# gets the rough size for a given module's parameters
def get_module_size( module ):
    param_size = sum([p.nelement() * p.element_size() for p in module.parameters()])
    buffer_size = sum([b.nelement() * b.element_size() for b in module.buffers()])
    return param_size + buffer_size

# to-do: rewrite all this shit, I don't know what I was thinking when implementing it this way
# it'd be better to just attach to layers itself rather than every single module

# assigns modules to requested devices for a given policy
def get_model_offload_policy(module, policy=None):
	# handle any other weird values this is set to
	if not isinstance(policy, dict):
		policy = {}
	
	# default to only include the core model, and not the other modules (embeddings) in the splitting policy
	if "include" not in policy:
		policy["include"] = ["model"]
	
	if "limits" not in policy:
		policy["limits"] = []

	if "assign" not in policy:
		policy["assign"] = []

	if "devices" not in policy:
		policy["devices"]  = get_devices() # + cpu to spill the remainder on CPU if overbudget

	# create initial device info
	devices = [ get_device_properties(device) | {"modules": []} for device in policy["devices"] ]
	modules = [ (name, get_module_size(module)) for name, module in module.named_modules() if not [*module.named_children()] and passes_policy( policy, name ) ]
	# filter
	modules = [ (name, size) for name, size in modules if name and size ]

	total_size = sum([size for name, size in modules])

	# set caps if requested in the policy
	for i, cap in enumerate(policy["limits"]):
		# no limit, skip
		if cap <= 0:
			continue
		# is fractional, scale to total size
		if cap < 1:
			cap = math.floor(total_size * cap)
		# available space is below cap, don't set
		if devices[i]["free"] < cap:
			continue
		# cap to requested size
		devices[i]["free"] = cap

	# assign if specific parts of the model are requested for assignment
	if policy["assign"]:
		discarded = []
		# yuck, there has to be a better way
		for device_index, includes in enumerate( policy["assign"] ):
			device = devices[device_index]

			buffered_modules = []
			buffered_size = device["free"]

			# iterate through list of modules to compare against includes
			for name, size in modules:
				# doesn't pass policy
				if not passes_policy( {"include": includes}, name ):
					continue
				# check if within budget
				if buffered_size - size >= 0:
					# add to buffer
					buffered_modules.append( (name, size) )
					buffered_size -= size
				# budget exceeded, flush buffer
				else:
					discarded += buffered_modules
					buffered_modules = []
					buffered_size = 0
					break

			if buffered_modules and buffered_size:
				device["modules"] += [ name for name, size in buffered_modules ]
				device["free"] = buffered_size

		modules = discarded

	device_index = 0
	module_index = 0
	# assign modules to each device
	while module_index < len(modules) and device_index < len(devices):
		device = devices[device_index]
		name, size = modules[module_index]

		# fits within budget
		if device["free"] - size >= 0:
			device["modules"].append( name )
			device["free"] -= size
			module_index += 1
		# does not fit in budget, increase device index
		else:
			device_index += 1
			_logger.info(f"Over budget for device: {device['name']}, shifting to next device: {name}, {size / (1024 ** 2)}MiB")

	# to-do: check that all modules are exhausted
	assert module_index >= len(modules)

	# only return devices with modules assigned
	return [ device for device in devices if device["modules"] ]

# handles naively splitting a model's layers across multiple devices
# this apparently works for training too? the test trainer seemed fine with it split between GPU and CPU
def offload_model( model, policy=None ):	
	policy = get_model_offload_policy(model, policy=policy)

	# move modules to respective devices
	for i, device in enumerate( policy ):
		# nothing assigned, skip
		if not device["modules"]:
			continue

		for name in device["modules"]:
			module = model.get_submodule(name)
			module = module.to( device["name"] )
			module.device = device['name']

	# wrap modules with forward to ensure all inputs are matched to its device
	for name, module in model.named_modules():
		if not hasattr( module, 'forward' ):
			continue

		module.forward = auto_align_inputs_forward(module)

	"""
	# Validate that the layers are all in the right spot
	for name, module in model.named_modules():
		if not not [*module.named_children()]:
			continue
		try:
			_logger.info( name, next(module.parameters()).device )
		except Exception as e:
			_logger.info( name, "?" )
			pass
	"""

	return model