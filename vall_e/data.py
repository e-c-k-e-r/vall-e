# todo: clean this mess up

import copy
import h5py
import json
import logging
import numpy as np
import os
import random
import torch
import itertools

from .config import cfg
from .emb.qnt import trim, trim_random, repeat_extend_audio, concat_audio, merge_audio, decode_to_file, decode as decode_qnt, encode as encode_qnt
from .utils.sampler import PoolSampler, OrderedSampler, BatchedOrderedSampler, RandomSampler
from .utils.distributed import global_rank, local_rank, world_size

from collections import defaultdict
from functools import cache, cached_property
from itertools import groupby, zip_longest
from pathlib import Path
from typing import Any

from torch import Tensor
from torch.utils.data import DataLoader, Dataset as _Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence

from tqdm.auto import tqdm
# torch.multiprocessing.set_sharing_strategy("file_system")

_logger = logging.getLogger(__name__)

# fold into a typical LLM sequence (one embedding rather than split embeddings)
def fold_inputs(
	text_list = [],
	lang_list = [],
	task_list = [],
	tone_list = [],
	prom_list = [],
	resp_list = [],
	targ_list = [],

	ignore_index = None,

	sep = 3,
	stop = 3,
	config = None,
	
	quant_levels = None,
):
	if config is None:
		config = cfg.model

	def _create_mask(l, device):
		seq = torch.arange(max(l), device=device).unsqueeze(0)  # (1 t)
		stop = torch.tensor(l, device=device).unsqueeze(1)  # (b 1)
		return (seq < stop).float()  # (b t)

	def list_to_tensor(x_list: list[Tensor]):
		l = list(map(len, x_list))
		x = pad_sequence(x_list).t()

		m = _create_mask(l, x_list[0].device)
		m = m.to(x)
		return x, m

	def process_prom_or_task(i, prom):
		if prom is None:
			return

		if isinstance(prom, str):
			task = get_task_symmap()[f'<{input}>']
			seq = torch.Tensor([task_start + task]).to(device=device, dtype=dtype)

			input_ids[i].append( seq )
			input_ids[i].append( sep )
			return

		# deinterleaved
		if quant_levels is not None:
			quant_level = quant_levels[i]
			if ignore_index is not None:
				seq = torch.Tensor( [ ignore_index for _ in range( prom.shape[0] ) ] ).to(device=device, dtype=dtype)
			else:
				seq = prom[:, quant_level].to(device=device, dtype=dtype).clone()
				for idx, token in enumerate( seq ):
					token += prom_start + ( config.audio_tokens * quant_level )
		# interleaved
		else:
			if ignore_index is not None:
				seq = torch.Tensor( [ ignore_index for _ in range( prom.shape[0] * prom.shape[1] ) ] ).to(device=device, dtype=dtype)
			else:
				seq = prom.flatten().to(device=device, dtype=dtype)
				for idx, token in enumerate( seq ):
					token += prom_start + ( config.audio_tokens * ( idx % config.resp_levels ) )

		input_ids[i].append( seq )
		input_ids[i].append( sep )

	"""
	if quant_levels is not None:
		resps_list = [ [] if l == 0 else resp for l, resp in zip(quant_levels, resp_list) ]
	"""

	device = text_list[0].device
	dtype = torch.int64

	batch_size = len(text_list)
	input_ids = [ [] for _ in range(batch_size) ]

	offset = 0
	
	sep = torch.Tensor([ sep ]).to(device=device, dtype=dtype)
	stop = torch.Tensor([ stop ]).to(device=device, dtype=dtype)

	text_start = 0
	text_end = text_start + config.text_tokens

	lang_start = text_end
	lang_end = lang_start + config.langs

	rvq_start = lang_end
	rvq_end = rvq_start + config.resp_levels

	prom_start = rvq_end
	prom_end = prom_start + config.audio_tokens * config.resp_levels

	task_start = prom_end
	task_end = task_start + config.tasks

	tone_start = task_end
	tone_end = tone_start + config.tones
	
	resp_start = tone_end
	resp_end = resp_start + config.audio_tokens * config.resp_levels

	# text tokens
	for i, text in enumerate(text_list):
		if isinstance(text, torch.Tensor):
			seq = text + text_start
		else:
			seq = torch.Tensor([text_start + text]).to(device=device, dtype=dtype)
		input_ids[i].append( seq )
		input_ids[i].append( sep )

	# lang tokens
	for i, lang in enumerate(lang_list):
		if isinstance(lang, torch.Tensor):
			seq = lang + lang_start
		else:
			seq = torch.Tensor([lang_start + lang]).to(device=device, dtype=dtype)
		input_ids[i].append( seq )
		input_ids[i].append( sep )
	
	# inject target quant_level
	if quant_levels is not None:
		for i, rvq in enumerate( quant_levels ):
			if isinstance(rvq, torch.Tensor):
				seq = rvq + rvq_start
			else:
				seq = torch.Tensor([rvq_start + rvq]).to(device=device, dtype=dtype)
			input_ids[i].append( seq )
			input_ids[i].append( sep )

	# prom / task tokens
	for i, prom in enumerate(prom_list):
		# list of proms with a possible task token
		if isinstance(prom, list):
			for p in prom:
				process_prom_or_task(i, p)
		# raw tensor
		else:
			process_prom_or_task(i, prom)

	# tone tokens
	for i, tone in enumerate(tone_list):
		if isinstance(tone, torch.Tensor):
			seq = tone + tone_start
		else:
			seq = torch.Tensor([tone_start + tone]).to(device=device, dtype=dtype)
		input_ids[i].append( seq )
		input_ids[i].append( sep )

	# resp tokens
	for i, resp in enumerate(resp_list):
		# deinterleaved
		if quant_levels is not None:
			# grab the previous rvq level
			quant_level = quant_levels[i] - 1
			# way to signal we want to inference for rvq level 0
			# without it, it's a random chance for any level to be selected again	
			if quant_level < 0:
				continue
			else:
				# my shitcode keeps things as lists of tensors for each level, so this handles it because lists can't index by tuples
				if isinstance(resp, list):
					seq = resp[quant_level].to(device=device, dtype=dtype).clone()
				else:
					seq = resp[:, quant_level].to(device=device, dtype=dtype).clone()

				for idx, token in enumerate( seq ):
					token += resp_start + ( config.audio_tokens * quant_level )

			input_ids[i].append( seq )
			input_ids[i].append( stop )
		# interleaved
		else:
			seq = resp.flatten().to(device=device, dtype=dtype)
			for idx, token in enumerate( seq ):
				token += resp_start + ( config.audio_tokens * ( idx % config.resp_levels ) )
		
			input_ids[i].append( seq )
			input_ids[i].append( stop )

	# targ list
	for i, resp in enumerate(targ_list):
		# deinterleaved
		if quant_levels is not None:
			quant_level = quant_levels[i]
			seq = resp[:, quant_level].to(device=device, dtype=dtype)
			for idx, token in enumerate( seq ):
				token += resp_start + ( config.audio_tokens * quant_level )
			
			input_ids[i].append( seq )
			input_ids[i].append( stop )
		# interleaved
		else:
			seq = resp.flatten().to(device=device, dtype=dtype)
			for idx, token in enumerate( seq ):
				token += resp_start + ( config.audio_tokens * ( idx % config.resp_levels ) )
		
			input_ids[i].append( seq )
			input_ids[i].append( stop )

	for i, batch in enumerate(input_ids):
		input_ids[i] = torch.concat(input_ids[i], dim=-1).to(device=device, dtype=dtype)

	return list_to_tensor(input_ids)

# unfold from one unified token ID space to separate token spaces
# to-do: unfold at a specific RVQ level instead if requested
def unfold_outputs(
	output_ids,

	sep = 3,
	stop = 3,
	
	config = None,
	quant_levels = None,
):
	def bin_to_rvqs( tokens ):
		length = len(tokens)
		"""
		if length % config.resp_levels == 0:
			tokens = torch.Tensor(tokens).reshape( config.resp_levels, length // config.resp_levels ).t()
		"""
		bins = [ [] for _ in range(config.resp_levels) ]
		for pos in range( length ):
			rvq = pos % config.resp_levels
			bins[rvq].append( tokens[pos] )
		nearest = ( len(bins) // config.resp_levels ) * config.resp_levels
		bins = bins[:nearest]
		return torch.Tensor(bins).t().to(device=device, dtype=dtype)

	if config is None:
		config = cfg.model

	device = output_ids.device
	dtype = torch.int64

	batch_size = output_ids.shape[0]

	text_list = [ [] for _ in range(batch_size) ]
	rvq_list  = [ [] for _ in range(batch_size) ]
	lang_list  = [ [] for _ in range(batch_size) ]
	task_list  = [ [] for _ in range(batch_size) ]
	tone_list  = [ [] for _ in range(batch_size) ]
	prom_list = [ [] for _ in range(batch_size) ]
	resp_list = [ [] for _ in range(batch_size) ]

	text_start = 0
	text_end = text_start + config.text_tokens

	lang_start = text_end
	lang_end = lang_start + config.langs

	rvq_start = lang_end
	rvq_end = rvq_start + config.resp_levels

	prom_start = rvq_end
	prom_end = prom_start + config.audio_tokens * config.resp_levels

	task_start = prom_end
	task_end = task_start + config.tasks

	tone_start = task_end
	tone_end = tone_start + config.tones
	
	resp_start = tone_end
	resp_end = resp_start + config.audio_tokens * config.resp_levels

	for i, batch in enumerate( output_ids ):
		# cringe logic to handle prefix resp for rvq levels > 0
		# a better way is to observe if the rvq level increased
		should_flush = False
		flushed = False
		for idx, token in enumerate( batch ):
			id = token.item()
			if id == sep or id == stop:
				if should_flush and quant_levels is not None and quant_levels[i] > 0:
					resp_list[i] = []
					should_flush = False
					flushed = True

				continue

			# text tokens
			if text_start <= id and id < text_end:
				text_list[i].append( (id - text_start) % config.text_tokens )
			# lang tokens
			elif lang_start <= id and id < lang_end:
				lang_list[i].append( (id - lang_start) % config.langs )
			# rvq levels
			elif rvq_start <= id and id < rvq_end:
				rvq_list[i].append( (id - rvq_start) % config.resp_levels )
			# prom tokens
			elif prom_start <= id and id < prom_end:
				prom_list[i].append( (id - prom_start) % config.audio_tokens )
			# task tokens
			elif task_start <= id and id < task_end:
				task_list[i].append( (id - task_start) % config.tasks )
			# lang tokens
			elif tone_start <= id and id < tone_end:
				tone_list[i].append( (id - tone_start) % config.tones )
			# resp tokens
			elif resp_start <= id and id < resp_end:
				resp_list[i].append( (id - resp_start) % config.audio_tokens )

				if not flushed:
					should_flush = True

		if quant_levels is not None:
			prom_list[i] = torch.Tensor(prom_list[i]).t().to(device=device, dtype=dtype)
			resp_list[i] = torch.Tensor(resp_list[i]).t().to(device=device, dtype=dtype)
		else:
			prom_list[i] = bin_to_rvqs( prom_list[i] )
			resp_list[i] = bin_to_rvqs( resp_list[i] )

		text_list[i] = torch.Tensor( text_list[i] ).to(device=device, dtype=dtype)
		task_list[i] = torch.Tensor( task_list[i] ).to(device=device, dtype=dtype)
		lang_list[i] = torch.Tensor( lang_list[i] ).to(device=device, dtype=dtype)
		tone_list[i] = torch.Tensor( tone_list[i] ).to(device=device, dtype=dtype)

	return dict(
		text_list=text_list,
		prom_list=prom_list,
		resp_list=resp_list,
		
		task_list=task_list,
		lang_list=lang_list,
		tone_list=tone_list,
	)

# to-do: clean up this symmap mess
def get_phone_symmap():
	return cfg.tokenizer.get_vocab()

def tokenize( phones ):
	return cfg.tokenizer.encode( "".join(phones) )

def get_lang_symmap():
	return {
		"en": 0,
		"ja": 1,
	}

def get_tone_symmap():
	return {
		"neutral": 0,
	}
	return symmap

def get_task_symmap():
	return {
		"<tts>": 0,
		"<tts-c>": 1,
		"<ns>": 2,
		"<sr>": 3,
		"<tse>": 4,
		"<soe>": 5,
		"<mask>": 6,
		"<eoe>": 7,

		"<nse>": 6, # fake
		"<cse>": 6, # fake
	}

def _replace_file_extension(path, suffix):
	return (path.parent / path.name.split(".")[0]).with_suffix(suffix)

def _get_quant_extension():
	return ".dac" if cfg.audio_backend == "dac" else ".enc"

def _get_phone_extension():
	return ".json" # if cfg.audio_backend == "dac" else ".phn.txt"

def _get_quant_path(path):
	return _replace_file_extension(path, _get_quant_extension())

def _get_phone_path(path):
	return _replace_file_extension(path, _get_phone_extension())

_durations_map = {}
# makeshift caching the above to disk
@cfg.diskcache()
def _get_duration_map( type="training" ):
	return _durations_map[type] if type in _durations_map else {}

@cfg.diskcache()
def _load_paths(dataset, type="training"):
	return { cfg.get_spkr( cfg.data_dir / data_dir / "dummy" ): _load_paths_from_metadata( data_dir, type=type, validate=cfg.dataset.validate and type == "training" ) for data_dir in tqdm(dataset, desc=f"Parsing dataset: {type}") }

def _load_paths_from_metadata(group_name, type="training", validate=False):
	data_dir = group_name if cfg.dataset.use_hdf5 else cfg.data_dir / group_name

	_fn = _get_hdf5_paths if cfg.dataset.use_hdf5 else _get_paths_of_extensions

	def key( id, entry=None ):
		return f"/{type}/{_get_hdf5_path(data_dir)}/{id}" if cfg.dataset.use_hdf5 else data_dir / id

	metadata_path = cfg.metadata_dir / f'{group_name}.json'
	metadata = {}

	if cfg.dataset.use_metadata and metadata_path.exists():
		metadata = json.loads(open( metadata_path, "r", encoding="utf-8" ).read())

	if len(metadata) == 0:
		return _fn( data_dir, type if cfg.dataset.use_hdf5 else _get_quant_extension(), validate )

	def _validate( id, entry ):
		phones = entry['phones'] if "phones" in entry else 0
		duration = entry['duration'] if "duration" in entry else 0

		# add to duration bucket
		k = key(id, entry)
		if type not in _durations_map:
			_durations_map[type] = {}
		_durations_map[type][k] = duration

		if not validate:
			return True

		return cfg.dataset.min_duration <= duration and duration <= cfg.dataset.max_duration

	return [ key(id, entry) for id, entry in metadata.items() if _validate(id, entry) ]


def _get_hdf5_path(path):
	# to-do: better validation
	#print(path)
	return str(path)

def _get_hdf5_paths( data_dir, type="training", validate=False ):
	data_dir = str(data_dir)
	
	key = f"/{type}/{_get_hdf5_path(data_dir)}"

	def _validate( id, entry ):
		phones = entry.attrs['phonemes']
		duration = entry.attrs['duration']

		if type not in _durations_map:
			_durations_map[type] = {}
		_durations_map[type][f"{key}/{id}"] = duration

		if not validate:
			return True
		
		return cfg.dataset.min_duration <= duration and duration <= cfg.dataset.max_duration

	return [ Path(f"{key}/{id}") for id, entry in cfg.hdf5[key].items() if _validate(id, entry) ] if key in cfg.hdf5 else []

def _get_paths_of_extensions( path, extensions=_get_quant_extension(), validate=False ):
	if isinstance(path, str):
		path = Path(path)

	def _validate(path):
		if "".join(path.suffixes) not in extensions:
			return False
		if not _get_phone_path(path).exists() or not _get_quant_path(path).exists():
			return False
		if not validate:
			return True
		# to-do: find an easy way to determine size from pickled quants without loading
		# to-do: find a consistent way to derive phoneme count from filesize (probably can't due to utf-8)
		phones = len(_get_phones(_get_phone_path(path))) # _get_phone_path(path).stat().st_size // 2 + 1
		return cfg.dataset.min_phones <= phones and phones <= cfg.dataset.max_phones


	return [ p for p in list(path.iterdir()) if _validate(p) ] if path.exists() and path.is_dir() else []

def _load_quants(path, return_metadata=False) -> Tensor:
	qnt = np.load(_get_quant_path(path), allow_pickle=True)[()]
	if return_metadata:
		return torch.from_numpy(qnt["codes"].astype(int))[0][:, :].t().to(torch.int16), qnt["metadata"]
	return torch.from_numpy(qnt["codes"].astype(int))[0][:, :].t().to(torch.int16)

# prune consecutive spaces
def _cleanup_phones( phones, targets=[" "]):
	return [ p for i, p in enumerate(phones) if p not in targets or ( p in targets and p != phones[i-1] ) ]

@cache
def _get_phones(path):
	phone_path = _get_phone_path(path)
	quant_path = _get_quant_path(path)
	if phone_path.exists():
		metadata = json.loads(open(phone_path, "r", encoding="utf-8").read())
	elif quant_path.exists():
		_, metadata = _load_quants( path, return_metadata=True )
	else:
		raise Exception(f"Could not load phonemes: {path}")

	content = metadata["phonemes"]
	return "".join(content)

def _interleaved_reorder(l, fn):
	groups = defaultdict(list)
	for e in l:
		groups[fn(e)].append(e)
	groups = {k: groups[k] for k in sorted(groups)}
	for interleaved in zip_longest(*groups.values()):
		for value in interleaved:
			if value is not None:
				yield value

class Dataset(_Dataset):
	def __init__(
		self,
		phone_symmap=None,
		training=False,
		extra_paths_by_spkr_name: dict[str, list] = {},
	):
		super().__init__()
		self._head = None
		self.sampler = None

		self.paths = []

		self.training = training
		self.dataset_type = "training" if self.training else "validation"
		self.dataset = cfg.dataset.training if self.training else cfg.dataset.validation
		self.sampler_type = cfg.dataset.sample_type if self.dataset_type == "training" else "path"
		self.sampler_order = cfg.dataset.sample_order
		self.sampler_shuffle = cfg.dataset.sample_shuffle

		# to-do: do not do validation if there's nothing in the validation
		# this just makes it be happy
		if len(self.dataset) == 0:
			self.dataset = cfg.dataset.training
		
		# dict of paths keyed by speaker names
		self.paths_by_spkr_name = _load_paths(self.dataset, self.dataset_type)

		# cull speakers if they do not have enough utterances
		if cfg.dataset.min_utterances > 0:
			keys = list(self.paths_by_spkr_name.keys())
			for key in keys:
				if len(self.paths_by_spkr_name[key]) < cfg.dataset.min_utterances:
					del self.paths_by_spkr_name[key]

		# flatten paths
		self.paths = list(itertools.chain.from_iterable(self.paths_by_spkr_name.values()))
		
		# split dataset accordingly per GPU
		if cfg.distributed and self.training:
			"""
			batches = len(self.paths) // world_size()
			start = batches * global_rank()
			end = batches * (global_rank() + 1)

			self.paths = self.paths[start:end]
			"""

			self.paths = [ path for i, path in enumerate(self.paths) if i % world_size() == 0 ]

			# recreate paths_by_spkr_name
			self.paths_by_spkr_name = {}
			for path in self.paths:
				name = cfg.get_spkr( Path(path) )
				if name not in self.paths_by_spkr_name:
					self.paths_by_spkr_name[name] = []
				self.paths_by_spkr_name[name].append( path )

		# do it here due to the above
		self.duration = 0
		self.duration_map = _get_duration_map( self.dataset_type )
		self.duration_buckets = {}

		# store in corresponding bucket
		for path in self.paths:
			duration = self.duration_map[path]
			self.duration += duration
			
			# only calc duration if we're going to order by duration
			if self.sampler_order != "duration":
				continue

			bucket = int(round(duration))
			if bucket not in self.duration_buckets:
				self.duration_buckets[bucket] = []
			self.duration_buckets[bucket].append( ( Path(path), duration ) )

		# ensure they're ordered
		self.duration_buckets = dict(sorted(self.duration_buckets.items()))

		# sort by duration
		if self.sampler_order == "duration":
			flattened = {}
			# sort and interleave
			for bucket in self.duration_buckets:
				# sort by duration
				self.duration_buckets[bucket].sort( key=lambda x: x[1] )
				# split to retain tuples
				flattened[bucket] = self.duration_buckets[bucket]
				# replace with path
				flattened[bucket] = [ x[0] for x in flattened[bucket] ]
				# flatten by paths
				flattened[bucket] = [*_interleaved_reorder(flattened[bucket], self.get_speaker)]
			# flatten paths
			self.paths = list(itertools.chain.from_iterable(flattened.values()))
		else:
			# just interleave
			self.paths = [*_interleaved_reorder(self.paths, self.get_speaker)]

		
		# dict of speakers keyed by speaker group
		self.spkrs_by_spkr_group = {}
		for data_dir in self.dataset:
			spkr = cfg.get_spkr( data_dir / "dummy" )
			spkr_group = cfg.get_spkr_group( data_dir / "dummy" )

			if spkr not in self.paths_by_spkr_name or len(self.paths_by_spkr_name[spkr]) < cfg.dataset.min_utterances:
				continue

			if spkr_group not in self.spkrs_by_spkr_group:
				self.spkrs_by_spkr_group[spkr_group] = []

			self.spkrs_by_spkr_group[spkr_group].append( spkr )

		self.spkr_groups = list(self.spkrs_by_spkr_group.keys())

		self.noise_paths = _load_paths(cfg.dataset.noise, "noise")
		self.noise_paths = list(itertools.chain.from_iterable(self.noise_paths.values()))

		self.phone_symmap = phone_symmap or self._get_phone_symmap()
		self.spkr_symmap = self._get_spkr_symmap()
		self.spkr_group_symmap = self._get_spkr_group_symmap()
		self.lang_symmap = self._get_lang_symmap()
		self.tone_symmap = self._get_tone_symmap()
		self.task_symmap = self._get_task_symmap()

		# grab IDs for bos, space, and eos for easy input creation later
		self.empty_text = [ cfg.tokenizer._bos_token, cfg.tokenizer.get_vocab()[" "], cfg.tokenizer._eos_token ]

		# have it fetch at training time if any is invalid, because the tokenizer obj might not have it easily fetchable ahead of itme
		# encoding before parallelizing things causes things to whine
		if self.empty_text[0] is None or self.empty_text[-1] is None:
			self.empty_text = None

		# assert len(self.phone_symmap) < 256, "Unique token count should be [0,255] to fit within uint8"
		self.text_dtype = torch.uint8 if len(self.phone_symmap) < 256 else torch.int16

		if len(self.paths) == 0:
			raise ValueError(f"No valid path is found for {self.dataset_type}")

		if self.sampler_type == "path":
			if self.sampler_order == "duration" and cfg.dataset.sample_max_duration_batch > 0:
				self.sampler = BatchedOrderedSampler(
					self.duration_buckets if not self.sampler_state_dict_path.exists() else {}, # pass nothing if we're just going to load from a state anyways
					max_duration=cfg.dataset.sample_max_duration_batch,
					max_batch_size=cfg.hyperparameters.batch_size if self.training else cfg.evaluation.batch_size,
					shuffle=self.sampler_shuffle
				)
			else:
				self.sampler = OrderedSampler( len(self) ) if not self.sampler_shuffle else RandomSampler( len(self) )
			self.samplers = {}
			self.spkr_samplers = {}
		else:
			self.sampler = RandomSampler( len(self) )
			self.samplers = { name: PoolSampler( paths, keep_all=True, shuffle=self.sampler_shuffle ) for name, paths in self.paths_by_spkr_name.items() }
			self.spkr_samplers = { name: PoolSampler( [*set(speakers)], keep_all=True, shuffle=self.sampler_shuffle ) for name, speakers in self.spkrs_by_spkr_group.items() }

		# loading validation state dict causes issues
		if self.dataset_type != "validation":
			self.load_state_dict()

	@cached_property
	def sampler_state_dict_path(self):
		return cfg.rel_path / f"sampler.{self.sampler_type}.rank{global_rank()}.pt"
		
	def get_speaker(self, path):
		if isinstance(path, str):
			path = Path(path)
		res = cfg.get_spkr(path)
		return res

	def get_speaker_group(self, path):
		if isinstance(path, str):
			path = Path(path)
		res = cfg.get_spkr_group(path)
		return res

	def get_language(self, speaker_group):
		lang = "en"
		for k, v in cfg.dataset.speaker_languages.items():
			if speaker_group in v:
				lang = k
				break

		return lang

	@cached_property
	def spkrs(self):
		return sorted({self.get_speaker(path) for path in self.paths})

	@cached_property
	def tasks(self):
		return cfg.dataset.tasks_list # ["tts", "tts", "ns", "sr", "tse", "tts", "tts"] # , "cse", "nse"

	def save_state_dict(self, path = None):
		if path is None:
			path = self.sampler_state_dict_path

		if self.sampler_type == "path":
			state_dict = self.sampler.get_state()
		else:
			state_dict = {
				"samplers": { name: sampler.get_state() for name, sampler in self.samplers.items() },
				"spkr_samplers": { name: sampler.get_state() for name, sampler in self.spkr_samplers.items() },
			}
		torch.save(state_dict, path)

	def load_state_dict(self, path = None):
		if path is None:
			path = self.sampler_state_dict_path

		if not path.exists():
			return

		state_dict = torch.load(path)
		if self.sampler_type == "path":
			state_dict = self.sampler.set_state(state_dict)
		else:
			for name, sampler in state_dict["samplers"].items():
				if name not in self.samplers:
					continue
				self.samplers[name].set_state( sampler )

			for name, sampler in state_dict["spkr_samplers"].items():
				if name not in self.spkr_samplers:
					continue
				self.spkr_samplers[name].set_state( sampler )

	def _get_phone_symmap(self):
		return get_phone_symmap()

	def _get_spkr_symmap(self):
		return {s: i for i, s in enumerate(self.spkrs)}

	def _get_spkr_group_symmap(self):
		return {s: i for i, s in enumerate(self.spkr_groups)}

	def _get_lang_symmap(self):
		return get_lang_symmap()

	def _get_tone_symmap(self):
		return get_tone_symmap()

	def _get_task_symmap(self):
		return get_task_symmap()

	def sample_noise(self):
		path = random.choice(self.noise_paths)

		if cfg.dataset.use_hdf5:
			key = _get_hdf5_path(path)
			qnt = torch.from_numpy(cfg.hdf5[key]["audio"][:, :]).to(torch.int16)
		else:
			qnt = _load_quants(path, return_metadata=False)
		return qnt

	def sample_speakers(self, ignore=[]):
		choices = set(self.spkrs) - set(ignore)
		return random.choice([*choices])

	def sample_utterance(self, spkr_name, ignore=[]):
		choices = [*(set(self.paths_by_spkr_name[spkr_name]) - set(ignore))]

		if len(choices) == 0:
			return None, None, None
		
		path = random.choice(choices)
			
		if cfg.dataset.use_hdf5:
			key = _get_hdf5_path(path)

			if key not in cfg.hdf5:
				raise RuntimeError(f'Key of Path ({path}) not in HDF5: {key}')

			metadata = cfg.hdf5[key].attrs

			text = cfg.hdf5[key]["text"][:]
			resps = cfg.hdf5[key]["audio"][:, :]
			
			text = torch.from_numpy(text).to(self.text_dtype)
			resps = torch.from_numpy(resps).to(torch.int16)
			
			"""
			lang = metadata["language"] if "language" in metadata else None
			tone = metadata["tone"] if "tone" in metadata else None
			"""
		else:
			resps, metadata = _load_quants(path, return_metadata=True)
			text = torch.tensor(tokenize( metadata["phonemes"] )).to(self.text_dtype)

			"""
			lang = metadata["language"] if "language" in metadata else None
			tone = metadata["tone"] if "tone" in metadata else None
			"""

		return path, text, resps


	def sample_prompts(self, spkr_name, ignore, should_trim=True):
		if not cfg.dataset.prompt_duration_range or cfg.dataset.prompt_duration_range[-1] == 0:
			return None

		prom_list = []

		choices = set(self.paths_by_spkr_name[spkr_name]) - {ignore}
		choices = [*choices]

		# no other utterances, it'd make more sense to prune speakers with only one utterance in the validation step
		if len(choices) == 0:
			choices = [*set(self.paths_by_spkr_name[spkr_name])]
			"""
			raise ValueError(
				f"Failed to find another different utterance for {spkr_name}."
			)
			"""

		prom_length = 0
		trim_length = int(random.uniform(cfg.dataset.prompt_duration_range[0], cfg.dataset.prompt_duration_range[1]) * cfg.dataset.frames_per_second) if trim else 0

		for _ in range(cfg.dataset.max_prompts):
			path = random.choice(choices)
			if cfg.dataset.use_hdf5:
				key = _get_hdf5_path(path)

				if "audio" not in cfg.hdf5[key]:
					_logger.warning(f'MISSING AUDIO: {key}')
					continue

				qnt = torch.from_numpy(cfg.hdf5[key]["audio"][:, :]).to(torch.int16)
			else:
				qnt = _load_quants(path, return_metadata=False)

			if 0 < trim_length and trim_length < qnt.shape[0]:
				qnt = trim( qnt, trim_length, reencode=cfg.dataset.reencode_on_concat, device=cfg.dataset.reencode_device )

			prom_list.append(qnt)
			prom_length += qnt.shape[0]

			if prom_length >= trim_length or random.random() > cfg.dataset.random_utterance:
				break

		# might be better to decode => concat waveforms with silence in between => reencode
		# as you technically can't just append encodec sequences together like this without issues
		prom = concat_audio( *prom_list, reencode=cfg.dataset.reencode_on_concat, device=cfg.dataset.reencode_device )

		if 0 < trim_length and trim_length < prom.shape[0]:
			prom = trim( prom, trim_length, reencode=cfg.dataset.reencode_on_concat, device=cfg.dataset.reencode_device )

		return prom

	def __getitem__(self, index):
		if self.empty_text is None:
			self.empty_text = tokenize(" ")
		
		bos_id, space_id, eos_id = self.empty_text

		if self.sampler_type == "group":
			spkr_group = self.spkr_groups[index]
			#spkr_group_id = self.spkr_group_symmap[spkr_group]
			spkr_name = self.spkr_samplers[spkr_group].sample()
			spkr_id = self.spkr_symmap[spkr_name]
			path = self.samplers[spkr_name].sample()
		elif self.sampler_type == "speaker":
			spkr_name = self.spkrs[index]
			spkr_id = self.spkr_symmap[spkr_name]
			path = self.samplers[spkr_name].sample()
			spkr_group = self.get_speaker_group(path)
			#spkr_group_id = self.spkr_group_symmap[spkr_group]
		else:
			path = self.paths[index]
			spkr_name = self.get_speaker(path)
			spkr_id = self.spkr_symmap[spkr_name]
			spkr_group = self.get_speaker_group(path)
			#spkr_group_id = self.spkr_group_symmap[spkr_group]

		if cfg.dataset.use_hdf5:
			key = _get_hdf5_path(path)

			if key not in cfg.hdf5:
				raise RuntimeError(f'Key of Path ({path}) not in HDF5: {key}')

			metadata = cfg.hdf5[key].attrs

			text = cfg.hdf5[key]["text"][:]
			resps = cfg.hdf5[key]["audio"][:, :]
			
			text = torch.from_numpy(text).to(self.text_dtype)
			resps = torch.from_numpy(resps).to(torch.int16)
			
			lang = metadata["language"] if "language" in metadata else None
			tone = metadata["tone"] if "tone" in metadata else None
			text_string = metadata["text"] if "text" in metadata else None
		else:
			resps, metadata = _load_quants(path, return_metadata=True)
			text = torch.tensor(tokenize( metadata["phonemes"] )).to(self.text_dtype)

			lang = metadata["language"] if "language" in metadata else None
			tone = metadata["tone"] if "tone" in metadata else None
			text_string = metadata["text"] if "text" in metadata else None

		if not lang:
			lang = self.get_language(spkr_group)
		
		if not tone:
			tone = "neutral"

		lang = torch.tensor([self.lang_symmap[lang]]).to(torch.uint8)
		tone = torch.tensor([self.tone_symmap[tone]]).to(torch.uint8)

		# a bool to easily experiment with two mindsets later
		naive = cfg.experimental

		# append additional prompts in an attempt to artifically increase lengths / offer new data
		if cfg.dataset.max_resps > 1 and random.random() < cfg.dataset.p_resp_append:
			ignore_paths = []
			for _ in range( 1, cfg.dataset.max_resps ):
				path, txt, qnt = self.sample_utterance(spkr_name, ignore=ignore_paths)
				ignore_paths.append(path)

				# <s>[original text]</s><s>[new text]</s>
				if naive:
					text = torch.concat([ text, txt ])
				# <s>[original text] [new text]</s>
				# removes the original text's </s>, includes a space, and remove the new text's <s>
				else:
					text = torch.concat([ text[:-1], torch.tensor([self.phone_symmap[" "]]).to(torch.int16),  txt[1:] ])

				# might be better to decode => concat waveforms with silence in between => reencode
				# as you technically can't just append encodec sequences together like this without issues
				resps = concat_audio( resps, qnt, reencode=cfg.dataset.reencode_on_concat, device=cfg.dataset.reencode_device )
		
		task = random.choice(self.tasks)

		if f'<{task}>' not in self.task_symmap:
			raise Exception(f'Task not defined: {task}')

		# Base TTS (<text><prompt> => <resp>)
		if task == "tts":
			proms = self.sample_prompts(spkr_name, ignore=path)

			if cfg.dataset.inject_noise_in_prom:
				# sample random noise
				noise = self.sample_noise()
				# extend the noise to fill the target audio
				noise = repeat_extend_audio(noise, proms.shape[0])
				# create the input prompt by merging the target audio with the noise
				proms = merge_audio( proms, noise, scale=[1, cfg.dataset.noise_scale], device=cfg.dataset.reencode_device )


		# VALL-E Continuous (<text><partial resp> => <remaining resp> )
		#     (this could just be sampled as <text a><text b><audio a> => <audio b>, but I need to experiment with it)
		elif task == "tts-c":
			# trim a piece of the output response
			if naive:
				trim_length = int(random.uniform(cfg.dataset.prompt_duration_range[0], cfg.dataset.prompt_duration_range[1]) * cfg.dataset.frames_per_second)
			
				proms = resps[:trim_length, :]
				resps = resps[trim_length:, :]
			else:
				path, txt, qnt = self.sample_utterance(spkr_name)

				# <s>[original text]</s><s>[new text]</s>
				if naive:
					text = torch.concat([ text, txt ])
				# <s>[original text] [new text]</s>
				# removes the original text's </s>, includes a space, and remove the new text's <s>
				else:
					text = torch.concat([ text[:-1], torch.tensor([space_id]).to(torch.int16), txt[1:] ])

				# set prompt as initial response
				proms = resps
				# set target as newly sampled response
				resps = qnt

			# inject task token
			proms = [
				proms,
				task,
			]

		# noise suppression (<text>? <resp+noise> => <resp>)
		# speech removal (<text>?<resp+noise> => <noise>)
		elif task == "ns" or task == "sr":
			# sample random noise
			noise = self.sample_noise()
			# extend the noise to fill the target audio
			noise = repeat_extend_audio(noise, resps.shape[0])
			# create the input prompt by merging the target audio with the noise
			proms = merge_audio( resps, noise, scale=[1, cfg.dataset.noise_scale], device=cfg.dataset.reencode_device )
			
			# set the text prompt to empty to train without a guided text prompt
			if random.random() < 0.5:
				text = None
			
			# inject task token
			proms = [
				task,
				proms
			]

			# set the target to just be the noise if <sr>
			if task == "sr":
				resps = noise


		# target speech extraction ( <text><prom><resp + other resp> => <resp> )
		elif task == "tse":
			# sample a prompt
			proms = self.sample_prompts(spkr_name, ignore=path)

			# sample another speaker
			_, __, other_resps = self.sample_utterance(self.sample_speakers(ignore=[spkr_name]))

			# overlay the random speaker over the target audio
			other_resps = merge_audio( resps, other_resps, scale=[1, random.uniform(0.5, 0.75)], device=cfg.dataset.reencode_device )

			# set the text prompt to empty to train without a guided text prompt
			if random.random() < 0.5:
				text = None

			# stitch together the proms
			proms = [
				proms,
				task,
				other_resps,
			]


		# clean speech editing
		elif task == "cse" or task == "nse":
			# speech editing would require higher quality transcription data (phoneme level/word level) unfortunately
			# as I need to get a good clean point to trim into
			# instead we'll just sample a bunch of utterances

			samples = []
			for _ in range( 4 ):
				sampled = self.sample_utterance(spkr_name, ignore=[s[0] for s in samples])
				samples.append( sampled )

			pre_text, mid_text, post_text, edit_text = [ s[1][1:-1] for s in samples ]
			pre_prom, mid_prom, post_prom, edit_prom = [ s[2] for s in samples ]

			# randomly drop out pre
			if random.random() < 0.125:
				pre_text = None
				pre_prom = None
			# randomly drop out post
			elif random.random() < 0.125:
				post_text = None
				post_prom = None

			# create new text
			text = concat_audio(
				torch.Tensor( [ bos_id ] ).to(dtype=self.text_dtype), # <s>
				pre_text,
				None if pre_text is None else torch.Tensor( [ space_id ] ).to(dtype=self.text_dtype), # " "
				edit_text,
				None if post_text is None else torch.Tensor( [ space_id ] ).to(dtype=self.text_dtype), # " "
				post_text,
				torch.Tensor( [ eos_id ] ).to(dtype=self.text_dtype), # </s>

				reencode=False,
			)

			if task == "nse":
				# sample random noise
				noise = self.sample_noise()

				# it might be better to extend the noise to the sum of the pre+mid+post or pre+edit+post to keep the noise truly coherent
				# but it's noise, it's supposed to be random
				def noise_proms( p ):
					# ignore if we turned it off
					if p is None:
						return None

					# extend the noise to fill the target audio
					n = repeat_extend_audio(noise, p.shape[0])
					# merge the noise over the utterance
					return merge_audio(p, n, scale=[1, cfg.dataset.noise_scale], device=cfg.dataset.reencode_device)
				
				# apply noise to all pieces
				pre_prom = noise_proms( pre_prom )
				mid_prom = noise_proms( mid_prom )
				post_prom = noise_proms( post_prom )
				edit_prom = noise_proms( edit_prom )

			# create new prom
			proms = [
				pre_prom,
				"soe",
				"mask" if task == "cse" else mid_prom,
				"eoe",
				post_prom,
			]

			# create new resp
			resps = concat_audio( 
				pre_prom,
				edit_prom,
				post_prom,
				reencode=cfg.dataset.reencode_on_concat,
				device=cfg.dataset.reencode_device,
			)
		else:
			raise Exception(f'Undefined task: {task}')

		if text is None:
			text = torch.tensor([bos_id, eos_id]).to(self.text_dtype)

		return dict(
			index=index,
			path=Path(path),
			spkr_name=spkr_name,
			spkr_id=spkr_id,
			task=task,
			lang=lang,
			tone=tone,
			text=text,
			proms=proms,
			resps=resps,
			
			metadata=metadata,
		)

	def head_(self, n):
		self._head = n

	def training_(self, value):
		self.training = value

	def __len__(self):
		if self.sampler_type == "group":
			return min(len(self.spkr_groups), self._head or len(self.spkr_groups))
		if self.sampler_type == "speaker":
			return min(len(self.spkrs), self._head or len(self.spkrs))
		return min(len(self.paths), self._head or len(self.paths))


def collate_fn(samples: list[dict]):
	batch: dict[str, Any] = {k: [s[k] for s in samples] for k in samples[0]}
	return batch


def _seed_worker(worker_id):
	worker_seed = torch.initial_seed() % 2**32
	np.random.seed(worker_seed)
	random.seed(worker_seed)


def _create_dataloader(dataset, training):
	kwargs = dict(
		shuffle=False,
		batch_size=cfg.hyperparameters.batch_size if training else cfg.evaluation.batch_size,
		drop_last=training,
		sampler=dataset.sampler if training else None,
	) if not isinstance(dataset.sampler, BatchedOrderedSampler) else dict(
		batch_sampler=dataset.sampler,
	)

	return DataLoader(
		dataset=dataset,
		num_workers=cfg.dataset.workers,
		collate_fn=collate_fn,
		persistent_workers=cfg.dataset.workers > 1,
		pin_memory=False,
		worker_init_fn=_seed_worker,
		**kwargs,
	)

def create_datasets():
	train_dataset = Dataset( training=True )
	val_dataset = Dataset( phone_symmap=train_dataset.phone_symmap, training=False )

	return train_dataset, val_dataset

def create_train_dataloader():
	train_dataset = Dataset( training=True )
	train_dl = _create_dataloader(train_dataset, training=True)

	_logger.info(str(train_dataset.phone_symmap))
	_logger.info(str(train_dataset.spkr_symmap))
	_logger.info(str(train_dataset.spkr_group_symmap))
	
	_logger.info(f"#samples (train): {len(train_dataset)}.")
	_logger.info(f"#duration (train): {str(train_dataset.duration)}.")

	return train_dl

def create_val_dataloader():
	val_dataset = Dataset( training=False )
	val_dl = _create_dataloader(val_dataset, training=False)

	_logger.info(str(val_dataset.phone_symmap))
	_logger.info(str(val_dataset.spkr_symmap))
	_logger.info(str(val_dataset.spkr_group_symmap))
	
	_logger.info(f"#samples (val): {len(val_dataset)}.")
	_logger.info(f"#duration (val): {str(val_dataset.duration)}.")

	return val_dl

def create_train_val_dataloader():
	train_dataset, val_dataset = create_datasets()

	# deepcopy is slow
	subtrain_dataset = Dataset( training=True ) 

	if subtrain_dataset.sampler_type == "path":
		subtrain_dataset.head_(cfg.evaluation.size)

	train_dl = _create_dataloader(train_dataset, training=True)
	val_dl = _create_dataloader(val_dataset, training=False)
	subtrain_dl = _create_dataloader(subtrain_dataset, training=False)

	_logger.info(str(train_dataset.phone_symmap))
	_logger.info(str(train_dataset.spkr_symmap))
	_logger.info(str(train_dataset.spkr_group_symmap))

	_logger.info(f"#samples (train): {len(train_dataset)}.")
	_logger.info(f"#samples (val): {len(val_dataset)}.")
	_logger.info(f"#samples (subtrain): {len(subtrain_dataset)}.")

	_logger.info(f"#duration (train): {str(train_dataset.duration)}.")
	_logger.info(f"#duration (val): {str(val_dataset.duration)}.")
	_logger.info(f"#duration (subtrain): {str(subtrain_dataset.duration)}.")

	assert isinstance(subtrain_dl.dataset, Dataset)

	return train_dl, subtrain_dl, val_dl

# parse dataset into better to sample metadata
def create_dataset_metadata( skip_existing=True ):
	symmap = get_phone_symmap()
	
	root = str(cfg.data_dir)
	metadata_root = str(cfg.metadata_dir)

	cfg.metadata_dir.mkdir(parents=True, exist_ok=True)

	def add( dir, type="training", audios=True, texts=True ):
		name = str(dir)
		name = name.replace(root, "")

		speaker_name = name

		metadata_path = Path(f"{metadata_root}/{speaker_name}.json")
		metadata_path.parents[0].mkdir(parents=True, exist_ok=True)

		try:
			metadata = {} if not metadata_path.exists() else json.loads(open(str(metadata_path), "r", encoding="utf-8").read())
		except Exception as e:
			metadata = {}

		if not os.path.isdir(f'{root}/{name}/'):
			return
		# tqdm.write(f'{root}/{name}')
		files = os.listdir(f'{root}/{name}/')

		# grab IDs for every file
		ids = { file.replace(_get_quant_extension(), "").replace(_get_phone_extension(), "") for file in files }

		for id in tqdm(ids, desc=f"Processing {name}"):
			try:
				quant_exists = os.path.exists(f'{root}/{name}/{id}{_get_quant_extension()}') if audios else True
				text_exists = os.path.exists(f'{root}/{name}/{id}{_get_phone_extension()}') if texts else True

				if not quant_exists:
					continue

				key = f'{type}/{speaker_name}/{id}'

				if skip_existing and id in metadata:
					continue

				if id not in metadata:
					metadata[id] = {}

				utterance_metadata = {}
				if audios:
					# ideally we'll encode Encodec-based audio in a similar manner because np has smaller files than pt
					dac = np.load(f'{root}/{name}/{id}{_get_quant_extension()}', allow_pickle=True)[()]
					qnt = torch.from_numpy(dac["codes"].astype(int))[0].t().to(dtype=torch.int16)

					if "text" in dac["metadata"]:
						utterance_metadata["text"] = dac["metadata"]["text"]
					if "phonemes" in dac["metadata"]:
						utterance_metadata["phonemes"] = dac["metadata"]["phonemes"]
					if "language" in dac["metadata"]:
						utterance_metadata["language"] = dac["metadata"]["language"]
					if "original_length" in dac["metadata"] and "sample_rate" in dac["metadata"]:
						utterance_metadata["duration"] = dac["metadata"]["original_length"] / dac["metadata"]["sample_rate"]
				# text
				if texts and text_exists and not utterance_metadata:
					utterance_metadata = json.loads(open(f'{root}/{name}/{id}{_get_phone_extension()}', "r", encoding="utf-8").read())

				for k, v in utterance_metadata.items():
					metadata[id][k] = v

			except Exception as e:
				tqdm.write(f'Error while processing {id}: {e}')

		with open(str(metadata_path), "w", encoding="utf-8") as f:
			f.write( json.dumps( metadata ) )

	# training
	for data_dir in tqdm(sorted(cfg.dataset.training), desc="Processing Training"):
		add( data_dir, type="training" )

	# validation
	for data_dir in tqdm(sorted(cfg.dataset.validation), desc='Processing Validation'):
		add( data_dir, type="validation" )

	# noise
	for data_dir in tqdm(sorted(cfg.dataset.noise), desc='Processing Noise'):
		add( data_dir, type="noise", texts=False )

# parse yaml to create an hdf5 file
def create_dataset_hdf5( skip_existing=True ):
	cfg.dataset.use_hdf5 = True
	cfg.load_hdf5(write=True)
	hf = cfg.hdf5

	symmap = get_phone_symmap()
	
	root = str(cfg.data_dir)
	metadata_root = str(cfg.metadata_dir)


	def add( dir, type="training", audios=True, texts=True ):
		name = str(dir)
		name = name.replace(root, "")
		
		# yucky
		speaker_name = name
		if "LibriTTS-R" in speaker_name:
			speaker_name = speaker_name.replace("LibriTTS-R", "LibriVox")

		metadata_path = Path(f"{metadata_root}/{speaker_name}.json")
		metadata_path.parents[0].mkdir(parents=True, exist_ok=True)

		metadata = {} if not metadata_path.exists() else json.loads(open(str(metadata_path), "r", encoding="utf-8").read())

		if not os.path.isdir(f'{root}/{name}/'):
			return

		files = os.listdir(f'{root}/{name}/')

		# grab IDs for every file
		ids = { file.replace(_get_quant_extension(), "").replace(_get_phone_extension(), "") for file in files }

		"""
		# rephonemizes if you fuck up and use and old tokenizer...
		for id, entry in tqdm(metadata.items(), desc=f"Processing {name}"):
			key = f'{type}/{speaker_name}/{id}'

			if key not in hf:
				continue
			
			group = hf[key]

			if "phonemes" not in entry:
				continue
			if "text" not in group:
				continue

			txt = entry["phonemes"]
			phn = "".join(txt)
			phn = cfg.tokenizer.encode(phn)
			phn = np.array(phn).astype(np.uint8) 

			del group["text"]
			group.create_dataset('text', data=phn, compression='lzf')
		"""

		for id in tqdm(ids, desc=f"Processing {name}"):
			try:
				quant_exists = os.path.exists(f'{root}/{name}/{id}{_get_quant_extension()}') if audios else True
				text_exists = os.path.exists(f'{root}/{name}/{id}{_get_phone_extension()}') if texts else True

				if not quant_exists:
					continue

				key = f'{type}/{speaker_name}/{id}'

				if skip_existing and key in hf:
					continue

				group = hf.create_group(key) if key not in hf else hf[key]

				if id not in metadata:
					metadata[id] = {}

				utterance_metadata = {}

				# audio
				if audios:
					dac = np.load(f'{root}/{name}/{id}{_get_quant_extension()}', allow_pickle=True)[()]
					qnt = torch.from_numpy(dac["codes"].astype(int))[0].t().to(dtype=torch.int16)

					if "text" in dac["metadata"]:
						utterance_metadata["text"] = dac["metadata"]["text"]
					if "phonemes" in dac["metadata"]:
						utterance_metadata["phonemes"] = dac["metadata"]["phonemes"]
					if "language" in dac["metadata"]:
						utterance_metadata["language"] = dac["metadata"]["language"]
					if "original_length" in dac["metadata"] and "sample_rate" in dac["metadata"]:
						utterance_metadata["duration"] = dac["metadata"]["original_length"] / dac["metadata"]["sample_rate"]

					if "audio" not in group:
						group.create_dataset('audio', data=qnt.numpy().astype(np.int16), compression='lzf')

				# text
				if texts:
					if not utterance_metadata and text_exists:
						utterance_metadata = json.loads(open(f'{root}/{name}/{id}{_get_phone_extension()}', "r", encoding="utf-8").read())

					phn = "".join(utterance_metadata["phonemes"])
					phn = cfg.tokenizer.encode(phn)
					phn = np.array(phn).astype(np.uint8) 

					if "text" not in group:
						group.create_dataset('text', data=phn, compression='lzf')

				for k, v in utterance_metadata.items():
					group.attrs[k] = v
					metadata[id][k] = v

			except Exception as e:
				tqdm.write(f'Error while processing {id}: {e}')

		"""
		with open(str(metadata_path), "w", encoding="utf-8") as f:
			f.write( json.dumps( metadata ) )
		"""


	# training
	for data_dir in tqdm(cfg.dataset.training, desc="Processing Training"):
		add( data_dir, type="training" )

	# validation
	for data_dir in tqdm(cfg.dataset.validation, desc='Processing Validation'):
		add( data_dir, type="validation" )

	# noise
	for data_dir in tqdm(cfg.dataset.noise, desc='Processing Noise'):
		add( data_dir, type="noise", texts=False )

	# write symmap
	if "symmap" in hf:
		del hf['symmap']

	hf.create_dataset('symmap', data=json.dumps(symmap))
	hf.close()

def transcribe_dataset():
	import os
	import json
	import torch
	import torchaudio
	import whisperx

	from tqdm.auto import tqdm
	from pathlib import Path

	# to-do: use argparser
	batch_size = 16
	device = "cuda" 
	dtype = "float16"
	model_name = "large-v3"

	input_audio = "voices"
	output_dataset = "training/metadata"

	skip_existing = True
	diarize = False

	# 
	model = whisperx.load_model(model_name, device, compute_type=dtype)
	align_model, align_model_metadata, align_model_language = (None, None, None)
	if diarize:
		diarize_model = whisperx.DiarizationPipeline(device=device)
	else:
		diarize_model = None

	def pad(num, zeroes):
		return str(num).zfill(zeroes+1)

	for dataset_name in os.listdir(f'./{input_audio}/'):
		if not os.path.isdir(f'./{input_audio}/{dataset_name}/'):
			continue

		for speaker_id in tqdm(os.listdir(f'./{input_audio}/{dataset_name}/'), desc="Processing speaker"):
			if not os.path.isdir(f'./{input_audio}/{dataset_name}/{speaker_id}'):
				continue

			outpath = Path(f'./{output_dataset}/{dataset_name}/{speaker_id}/whisper.json')

			if outpath.exists():
				metadata = json.loads(open(outpath, 'r', encoding='utf-8').read())
			else:
				os.makedirs(f'./{output_dataset}/{dataset_name}/{speaker_id}/', exist_ok=True)
				metadata = {}

			for filename in tqdm(os.listdir(f'./{input_audio}/{dataset_name}/{speaker_id}/'), desc=f"Processing speaker: {speaker_id}"):

				if skip_existing and filename in metadata:
					continue

				if ".json" in filename:
					continue

				inpath = f'./{input_audio}/{dataset_name}/{speaker_id}/{filename}'

				if os.path.isdir(inpath):
					continue
				
				metadata[filename] = {
					"segments": [],
					"language": "",
					"text": "",
					"start": 0,
					"end": 0,
				}

				audio = whisperx.load_audio(inpath)
				result = model.transcribe(audio, batch_size=batch_size)
				language = result["language"]

				if language[:2] not in ["ja"]:
					language = "en"

				if align_model_language != language:
					tqdm.write(f'Loading language: {language}')
					align_model, align_model_metadata = whisperx.load_align_model(language_code=language, device=device)
					align_model_language = language

				result = whisperx.align(result["segments"], align_model, align_model_metadata, audio, device, return_char_alignments=False)

				metadata[filename]["segments"] = result["segments"]
				metadata[filename]["language"] = language

				if diarize_model is not None:
					diarize_segments = diarize_model(audio)
					result = whisperx.assign_word_speakers(diarize_segments, result)

				text = []
				start = 0
				end = 0
				for segment in result["segments"]:
					text.append( segment["text"] )
					start = min( start, segment["start"] )
					end = max( end, segment["end"] )

				metadata[filename]["text"] = " ".join(text).strip()
				metadata[filename]["start"] = start
				metadata[filename]["end"] = end

				open(outpath, 'w', encoding='utf-8').write(json.dumps(metadata))

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser("Save trained model to path.")
	parser.add_argument("--action", type=str)
	parser.add_argument("--tasks", type=str)
	args, unknown = parser.parse_known_args()

	task = args.action

	cfg.dataset.workers = 1

	class LoggerOveride:
		def info(self, *args):
			print(*args)
	
	_logger = LoggerOveride()

	if args.action == "hdf5":
		transcribe_dataset()
	elif args.action == "hdf5":
		create_dataset_hdf5()
	elif args.action == "list-dataset":
		dataset = []
		for group in os.listdir(cfg.data_dir):
			for name in os.listdir(cfg.data_dir / group):
				if len(os.listdir(cfg.data_dir / group / name)) == 0:
					continue
				dataset.append(f'{group}/{name}')

		print(json.dumps(dataset))
	elif args.action == "metadata":
		create_dataset_metadata()
	elif args.action == "sample":
		train_dl, subtrain_dl, val_dl = create_train_val_dataloader()

		samples = {
			"training": [ next(iter(train_dl)),  next(iter(train_dl)) ],
			"evaluation": [ next(iter(subtrain_dl)),  next(iter(subtrain_dl)) ],
			#"validation": [ next(iter(val_dl)),  next(iter(val_dl)) ],
		}

		Path("./data/sample-test/").mkdir(parents=True, exist_ok=True)

		for k, v in samples.items():
			for i in range(len(v)):
				for j in tqdm(range(len(v[i]['proms'])), desc="Decoding..."):
					"""
					try:
						decode_to_file( v[i]['proms'][j], f"./data/sample-test/{k}.{i}.{j}.proms.wav", device="cpu" )
					except Exception as e:
						print(f"Error while decoding prom {k}.{i}.{j}.wav:", str(e))
					try:
						decode_to_file( v[i]['resps'][j], f"./data/sample-test/{k}.{i}.{j}.resps.wav", device="cpu" )
					except Exception as e:
						print(f"Error while decoding resp {k}.{i}.{j}.wav:", str(e))
					"""
					v[i]['proms'][j] = v[i]['proms'][j].shape
					v[i]['resps'][j] = v[i]['resps'][j].shape
		
		for k, v in samples.items():
			for i in range(len(v)):
				print(f'{k}[{i}]:', v[i])
	elif args.action == "validate":
		train_dl, subtrain_dl, val_dl = create_train_val_dataloader()

		missing = set()

		for i in range(len( train_dl.dataset )):
			batch = train_dl.dataset[i]

			text = batch['text']
			phonemes = batch['metadata']['phonemes']

			decoded = [ cfg.tokenizer.decode(token) for token in text[1:-1] ]
			for i, token in enumerate(decoded):
				if token != "<unk>":
					continue
				
				phone = phonemes[i]

				print( batch['text'], batch['metadata']['phonemes'] )

				missing |= set([phone])

		print( "Missing tokens:", missing )


	elif args.action == "tasks":
		index = 0
		cfg.dataset.tasks_list = args.tasks.split(",")
		
		train_dl, subtrain_dl, val_dl = create_train_val_dataloader()
		batch = next(iter(train_dl))

		for text, resps, proms, task in zip(batch["text"], batch["resps"], batch["proms"], batch["task"]):
			if task not in cfg.dataset.tasks_list:
				continue

			print(text, task, cfg.model.resp_levels)
			print( proms.shape, resps.shape )

			tokens = 0
			tokens += sum([ text.shape[0] for text in batch["text"] ])
			tokens += sum([ resps.shape[0] for resps in batch["resps"] ])
			print( tokens )

			decode_to_file( proms, f"./data/{task}.proms.wav", device="cpu" )
			decode_to_file( resps, f"./data/{task}.resps.wav", device="cpu" )
			break