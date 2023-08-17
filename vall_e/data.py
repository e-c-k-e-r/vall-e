# todo: clean this mess up

import copy
import h5py
import json
import logging
import numpy as np
import os
import random
import torch

from .config import cfg
from .utils.sampler import Sampler

from collections import defaultdict
from functools import cache, cached_property
from itertools import groupby, zip_longest
from pathlib import Path
from typing import Any

from torch import Tensor
from torch.utils.data import DataLoader, Dataset as _Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

# torch.multiprocessing.set_sharing_strategy("file_system")

_logger = logging.getLogger(__name__)

def get_phone_symmap():
	if cfg.dataset.use_hdf5 and 'symmap' in cfg.hdf5:
		return json.loads( cfg.hdf5['symmap'].asstr()[()] )

	symmap = {'<s>': 1, '</s>': 2, ' ': 3, '.': 4, ',': 5, '!': 6, '?': 7, 'p': 7, 'iː': 8, 'ɚ': 9, 'ˌ': 10, 'dˌ': 11, 'mˌ': 12, 'd': 13, 'ɹ': 14, 'tˈ': 15, 'pˌ': 16, 'uː': 17, 'l': 18, 'æ': 19, 'ɛ': 20, 'ɪ': 21, 'j': 22, 'ʊ': 23, 't': 24, 'n': 25, 'v': 26, 'a': 27, 'o': 28, 'ŋ': 29, 'w': 30, 'ʌ': 31, 'hˈ': 32, 'ɡˈ': 33, 'ə': 34, 'θˈ': 35, 'dˈ': 36, 'wˌ': 37, 'h': 38, 'z': 39, 'k': 40, 'ð': 41, 'ɡˌ': 42, 'ˈ': 43, 'fˈ': 44, 'i': 45, 's': 46, 'ʃ': 47, 'wˈ': 48, 'ðˈ': 49, 'ɹˈ': 50, 'lˈ': 51, 'ɡ': 52, 'oː': 53, 'mˈ': 54, 'e': 55, 'ɑː': 56, 'nˈ': 57, 'm': 58, 'θˌ': 59, 'sˈ': 60, 'f': 61, 'ɔː': 62, 'hˌ': 63, 'b': 64, 'jˈ': 65, 'ɐ': 66, 'ʒˈ': 67, 'θ': 68, 'bˈ': 69, 'ɾ': 70, 'ɜː': 71, 'ʌˈ': 72, 'ʃˌ': 73, 'bˌ': 74, 'kˈ': 75, 'ɔ': 76, 'zˈ': 77, 'ᵻ': 78, 'kˌ': 79, 'vˈ': 80, 'fˌ': 81, 'ʒ': 82, 'ʃˈ': 83, 'ɹˌ': 84, 'tˌ': 85, 'pˈ': 86, 'ðˌ': 87, 'sˌ': 88, 'nˌ': 89, 'lˌ': 90, '̩': 91, 'ʔ': 92, 'vˌ': 93, 'ɪˈ': 94, '"': 95, 'ɪˌ': 96, 'ʒˌ': 97, 'uːˌ': 98, 'ʊˈ': 99, 'jˌ': 100, 'uːˈ': 101, 'iːˈ': 102, 'zˌ': 103, '.ˈ': 104, '…': 105, 'ŋˌ': 106, 'ɐˌ': 107, '—ˈ': 108, 'iˌ': 109, 'iːˌ': 110, 'ɛː': 111, ')': 112, ')ˈ': 113, '(': 114, 'u': 115, '-': 116, 'ɖˈ': 117, 'iˈ': 118, 'ʰˈ': 119, 'ɟˈ': 120, '̃': 121, 'eː': 122, 'ɾˈ': 123, 'r': 124, 'ʰ': 125, '-ˌ': 126, 'ɫ': 127, 'q': 128, '—': 129, 'ʊˌ': 130, 'aː': 131, 'cˈ': 132, '…ˈ': 133, 'c': 134, 'ɳ': 135, 'ɐˈ': 136, 'x': 137, 'ʔˌ': 138, '.ˌ': 139, 'ɑ': 140, '?ˈ': 141, '̩ˈ': 142, '"ˈ': 143, ',ˈ': 144, 'ŋˈ': 145, 'əˌ': 146, '!ˈ': 147, '"ˌ': 148, '?ˌ': 149, ',ˌ': 150, '—ˌ': 151, '̩ˌ': 152, 'əˈ': 153, '!ˌ': 154, 'ɬ': 155, 'ʲ': 156, '¡': 157, 'ɯ': 158, 'qˌ': 159, 'ʑ': 160, 'ʑˈ': 161, '¿': 162, 'ɑːˈ': 163, 'iːː': 164, 'ɛˈ': 165, '¡ˈ': 166, 'æˈ': 167, 'ç': 168, 'ɾˌ': 169, 'ᵻˈ': 170, 'xˈ': 171, 'ɔːˈ': 172, ';': 173, 'ɬˌ': 174, ':': 175, 'ʔˈ': 176, 'ɑːˌ': 177, 'ɬˈ': 178}
	return symmap

def _replace_file_extension(path, suffix):
	return (path.parent / path.name.split(".")[0]).with_suffix(suffix)

def _get_hdf5_path(path):
	path = str(path)
	if path[:2] != "./":
		path = f'./{path}'
	return path.replace(cfg.cfg_path, "")

def _get_quant_path(path):
	return _replace_file_extension(path, ".qnt.pt")

def _get_phone_path(path):
	return _replace_file_extension(path, ".phn.txt")


def _load_quants(path) -> Tensor:
	path = _get_quant_path(path)
	return torch.load(path)[0][:cfg.models.levels, :].t().to(torch.int16)


@cache
def _get_phones(path, lang_marker="en"):
	path = _get_phone_path(path)
	with open(path, "r", encoding="utf8") as f:
		content = f.read()
	split = content.split(" ")
	return [f"<s>"] + [ " " if not p else p for p in split ] + [f"</s>"]


def _interleaved_reorder(l, fn):
	groups = defaultdict(list)
	for e in l:
		groups[fn(e)].append(e)
	groups = {k: groups[k] for k in sorted(groups)}
	for interleaved in zip_longest(*groups.values()):
		for value in interleaved:
			if value is not None:
				yield value


@cache
def _validate(path, min_phones, max_phones, min_duration, max_duration):
	if cfg.dataset.use_hdf5:
		key = _get_hdf5_path(path)
		if key not in cfg.hdf5:
			return False

		phones = cfg.hdf5[key].attrs['phonemes']
		duration = cfg.hdf5[key].attrs['duration']

		if phones < min_phones or phones > max_phones:
		   return False

		if duration < min_duration or duration > max_duration:
		   return False

		return True

	if not os.path.exists(_get_phone_path(path)) or not os.path.exists(_get_quant_path(path)):
		return False

	phones = _get_phones(path)
	unique_phones = list(set(phones))

	if len(unique_phones) == 0:
		return False
	if len(unique_phones) == 1 and unique_phones[0] == " ":
		return False
	if len(phones) < min_phones or len(phones) > max_phones:
		return False
	return True

class Dataset(_Dataset):
	def __init__(
		self,
		paths,
		phone_symmap=None,
		spkr_symmap=None,
		min_phones=cfg.dataset.phones_range[0],
		max_phones=cfg.dataset.phones_range[1],
		min_duration=cfg.dataset.duration_range[0],
		max_duration=cfg.dataset.duration_range[1],
		training=False,
		extra_paths_by_spkr_name: dict[str, list] = {},
		sample_type=cfg.dataset.sample_type # path | speaker
	):
		super().__init__()
		self._head = None
		self.min_phones = min_phones
		self.max_phones = max_phones
		self.min_duration = min_duration
		self.max_duration = max_duration
		self.sample_type = sample_type

		if cfg.dataset.validate:
			self.paths = [
				path for path in paths if _validate(path, self.min_phones, self.max_phones, self.min_duration, self.max_duration)
			]
		else:
			self.paths = paths

		self.spkr_symmap = spkr_symmap or self._get_spkr_symmap()
		self.phone_symmap = phone_symmap or self._get_phone_symmap()
		self.training = training

		# assert len(self.phone_symmap) < 256, "Unique token count should be [0,255] to fit within uint8"
		self.text_dtype = torch.uint8 if len(self.phone_symmap) < 256 else torch.int16

		self.paths_by_spkr_name = self._get_paths_by_spkr_name(extra_paths_by_spkr_name)

		if cfg.dataset.validate:
			self.paths = [
				p for p in self.paths if len(self.paths_by_spkr_name[cfg.get_spkr(p)]) > 1
			]

		if len(self.paths) == 0 and training:
			raise ValueError("No valid path is found for training.")
		
		self.duration = 0
		self.durations = {}
		if cfg.dataset.use_hdf5:
			for path in self.paths:
				key = _get_hdf5_path(path)
				spkr_name = cfg.get_spkr(path)
				spkr_id = self.spkr_symmap[spkr_name]
				duration = cfg.hdf5[key].attrs['duration']
				
				self.duration += duration

				if spkr_id not in self.durations:
					self.durations[spkr_id] = duration
				else:
					self.durations[spkr_id] += duration

		if training and not cfg.distributed and self.sample_type == "path":
			self.sampler = Sampler(self.paths, [cfg.get_spkr])
		else:
			self.sampler = None

	def _get_paths_by_spkr_name(self, extra_paths_by_spkr_name: dict[str, list]):
		ret = defaultdict(list)
		for path in self.paths:
			ret[cfg.get_spkr(path)].append(path)
		for k, v in extra_paths_by_spkr_name.items():
			ret[k].extend(v)
		return {**ret}

	@cached_property
	def phones(self):
		return sorted(set().union(*[_get_phones(path) for path in self.paths]))

	def _get_phone_symmap(self):
		return get_phone_symmap()

	@cached_property
	def spkrs(self):
		return sorted({cfg.get_spkr(path) for path in self.paths})

	def _get_spkr_symmap(self):
		return {s: i for i, s in enumerate(self.spkrs)}

	def sample_prompts(self, spkr_name, ignore):
		prom_list = []

		choices = set(self.paths_by_spkr_name[spkr_name]) - {ignore}
		choices = [*choices]

		# no other utterances, it'd make more sense to prune speakers with only one utterance in the validatoin step
		if len(choices) == 0:
			choices = [*set(self.paths_by_spkr_name[spkr_name])]
			"""
			raise ValueError(
				f"Failed to find another different utterance for {spkr_name}."
			)
			"""

		# shuffle it up a bit
		offset = random.randint(-16, 16)
		trim_length = int(cfg.dataset.prompt_duration * 75) + offset
		def trim( qnt ):
			length = qnt.shape[0]
			start = int(length * random.random())
			end = start + trim_length
			if end >= length:
				start = length - trim_length
				end = length                

			return qnt[start:end]
		
		total_qnt_length = 0
		for _ in range(cfg.dataset.max_prompts):
			path = random.choice(choices)
			if cfg.dataset.use_hdf5:
				key = _get_hdf5_path(path)
				#qnt = torch.from_numpy(cfg.hdf5[key]["audio"][:]).to(torch.int16)
				qnt = torch.from_numpy(cfg.hdf5[key]["audio"][:, :cfg.models.levels]).to(torch.int16)
			else:
				qnt = _load_quants(path)

			if cfg.dataset.prompt_duration > 0 and trim_length < qnt.shape[0]:
				qnt = trim(qnt)

			prom_list.append(qnt)
			total_qnt_length += qnt.shape[0]

			if total_qnt_length >= trim_length:
				break

			if random.random() > cfg.dataset.random_utterance:
				break

		prom = torch.cat(prom_list)

		if cfg.dataset.prompt_duration > 0 and trim_length < prom.shape[0]:
			prom = trim(prom)

		return prom

	@cached_property
	def tasks(self):
		return ["tts"] # "ns", "sr", "tse", "cse", "nse"

	def __getitem__(self, index):
		if hasattr(self, "sample_type") and self.sample_type == "speaker":
			spkr_name = self.spkrs[index]
			spkr_id = self.spkr_symmap[spkr_name]
			path = random.choice([*set(self.paths_by_spkr_name[spkr_name])])
		else:
			if self.training and self.sampler is not None:
				path = self.sampler.sample()
			else:
				path = self.paths[index]
			spkr_name = cfg.get_spkr(path)
			spkr_id = self.spkr_symmap[spkr_name]

		if cfg.dataset.use_hdf5:
			key = _get_hdf5_path(path)
			text = torch.from_numpy(cfg.hdf5[key]["text"][:]).to(self.text_dtype)
			resps = torch.from_numpy(cfg.hdf5[key]["audio"][:, :cfg.models.levels]).to(torch.int16)
		else:
			text = torch.tensor([*map(self.phone_symmap.get, _get_phones(path))]).to(self.text_dtype)
			resps = _load_quants(path)
		
		task = random.choice(self.tasks)
		if task == "tts":
			# I could probably do some logic to directly use the resps, but I'm putting my faith in python aliasing
			proms = self.sample_prompts(spkr_name, ignore=path) if random.random() < cfg.dataset.random_utterance else resps

		return dict(
			index=index,
			path=path,
			spkr_name=spkr_name,
			spkr_id=spkr_id,
			task=task,
			text=text,
			proms=proms,
			resps=resps,
		)

	def head_(self, n):
		self._head = n

	def training_(self, value):
		self.training = value

	def interleaved_reorder_(self, fn):
		self.paths = [*_interleaved_reorder(self.paths, fn)]

	def __len__(self):
		if hasattr(self, "sample_type") and self.sample_type == "speaker":
			return min(len(self.spkrs), self._head or len(self.spkrs))
		return min(len(self.paths), self._head or len(self.paths))

	def pin_memory(self):
		self.text = self.text.pin_memory()
		self.proms = self.proms.pin_memory()
		self.resps = self.resps.pin_memory()
		self.resp = self.resp.pin_memory()
		return self


def collate_fn(samples: list[dict]):
	batch: dict[str, Any] = {k: [s[k] for s in samples] for k in samples[0]}
	return batch


def _seed_worker(worker_id):
	worker_seed = torch.initial_seed() % 2**32
	np.random.seed(worker_seed)
	random.seed(worker_seed)


def _create_dataloader(dataset, training):
	sampler = None
	shuffle = True

	if cfg.distributed and training:
		sampler = DistributedSampler(dataset)
		shuffle = False

	return DataLoader(
		dataset=dataset,
		batch_size=cfg.hyperparameters.batch_size if training else cfg.evaluation.batch_size,
		shuffle=shuffle,
		drop_last=training,
		num_workers=cfg.dataset.workers,
		collate_fn=collate_fn,
		persistent_workers=True,
		pin_memory=False, # True,
		worker_init_fn=_seed_worker,
		sampler=sampler,
	)

def _load_dataset_paths():
	hf = cfg.hdf5
	paths = {
		"training": [],
		"validation": [],
	}

	datasets = {
		"training": [],
		"validation": [],
	}

	def get_paths( data_dir, type="training" ):
		key = f"/{type}{_get_hdf5_path(data_dir)}"
		if key not in cfg.hdf5:
			return

		paths[type].extend([ f"{key}/{child.attrs['id']}" for child in cfg.hdf5[key].values() ])

	for data_dir in cfg.dataset.training:
		get_paths( data_dir, "training" )

	for data_dir in cfg.dataset.validation:
		get_paths( data_dir, "validation" )

	for _, type in enumerate(paths):
		dirs = paths[type]

		if len(dirs) == 0:
			continue

		dirs = [ Path(p) for p in dirs ]

		pairs = sorted([(cfg.get_spkr(p), p) for p in dirs])
		for _, group in groupby(pairs, lambda pair: pair[0]):
			shuffled = sorted([p for _, p in group])
			random.seed(0)
			random.shuffle(shuffled)

			datasets[type].extend(shuffled)

	return datasets["training"], datasets["validation"]

# to-do: mirror the hdf5-based load function
def _load_train_val_paths():
	paths = []
	train_paths = []
	val_paths = []

	for data_dir in cfg.dataset.training:
		paths.extend(data_dir.rglob("*.qnt.pt"))

	if len(paths) > 0:
		pairs = sorted([(cfg.get_spkr(p), p) for p in paths])
		del paths

		for _, group in groupby(pairs, lambda pair: pair[0]):
			paths = sorted([p for _, p in group])
			random.seed(0)
			random.shuffle(paths)
			train_paths.extend(paths)

	for data_dir in cfg.dataset.validation:
		paths.extend(data_dir.rglob("*.qnt.pt"))

	if len(paths) > 0:
		pairs = sorted([(cfg.get_spkr(p), p) for p in paths])
		del paths

		for _, group in groupby(pairs, lambda pair: pair[0]):
			paths = sorted([p for _, p in group])
			random.seed(0)
			random.shuffle(paths)
			val_paths.extend(paths)

	train_paths, val_paths = map(sorted, [train_paths, val_paths])

	if len(train_paths) == 0:
		raise RuntimeError(f"Failed to find any .qnt.pt file in specified training dataset.")
	
	# to-do: allow setting aside a fixed portion of the training dataset for validation
	# something like the last X percent of each speaker is set aside
	if len(val_paths) == 0:
		val_paths = [ train_paths[0] ]

	return train_paths, val_paths

@cfg.diskcache()
def create_datasets():
	train_paths, val_paths = _load_dataset_paths() if cfg.dataset.use_hdf5 else _load_train_val_paths()

	train_dataset = Dataset(
		train_paths,
		training=True,
	)

	val_dataset = Dataset(
		val_paths,
		train_dataset.phone_symmap,
		#train_dataset.spkr_symmap,
		#extra_paths_by_spkr_name=train_dataset.paths_by_spkr_name,
	)

	val_dataset.interleaved_reorder_(cfg.get_spkr)
	val_dataset.head_(cfg.evaluation.size)

	return train_dataset, val_dataset


def create_train_val_dataloader():
	train_dataset, val_dataset = create_datasets()
	train_dataset.sample_type = cfg.dataset.sample_type #"speaker"

	subtrain_dataset = copy.deepcopy(train_dataset)
	if subtrain_dataset.sample_type == "path":
		subtrain_dataset.head_(cfg.evaluation.size)
		subtrain_dataset.interleaved_reorder_(cfg.get_spkr)

	train_dl = _create_dataloader(train_dataset, training=True)
	val_dl = _create_dataloader(val_dataset, training=False)
	subtrain_dl = _create_dataloader(subtrain_dataset, training=False)

	_logger.info(str(train_dataset.phone_symmap))
	_logger.info(str(train_dataset.spkr_symmap))
	

	_logger.info(f"#samples (train): {len(train_dataset)}.")
	_logger.info(f"#samples (val): {len(val_dataset)}.")
	_logger.info(f"#samples (subtrain): {len(subtrain_dataset)}.")
	
	"""
	_logger.info(f"#durations (train): {str(train_dataset.durations)}.")
	_logger.info(f"#durations (val): {str(val_dataset.durations)}.")
	_logger.info(f"#durations (subtrain): {str(subtrain_dataset.durations)}.")
	"""

	_logger.info(f"#duration (train): {str(train_dataset.duration)}.")
	_logger.info(f"#duration (val): {str(val_dataset.duration)}.")
	_logger.info(f"#duration (subtrain): {str(subtrain_dataset.duration)}.")

	assert isinstance(subtrain_dl.dataset, Dataset)

	return train_dl, subtrain_dl, val_dl

# parse yaml to create an hdf5 tile
def create_dataset_hdf5():
	symmap = get_phone_symmap()
	
	root = cfg.cfg_path
	hf = cfg.hdf5

	def add( dir, type="training" ):
		dir = "./" + str(dir)
		name = dir.replace(root, "")

		print( str(dir), name )

		if not os.path.isdir(f'{root}/{name}/'):
			return
		# tqdm.write(f'{root}/{name}')
		files = os.listdir(f'{root}/{name}/')

		# grab IDs for every file
		ids = { ".".join(file.split(".")[:-2]) for file in files }
		for id in tqdm(ids, desc=f"Processing {name}"):
			if not os.path.exists(f'{root}/{name}/{id}.qnt.pt') or not os.path.exists(f'{root}/{name}/{id}.phn.txt'):
				continue

			key = f'{type}/{name}/{id}'
			if key in hf:
				# print("Skipping existing entry:", key)
				continue

			group = hf.create_group(key)

			# audio
			qnt = torch.load(f'{root}/{name}/{id}.qnt.pt')[0].t()
			group.create_dataset('audio', data=qnt.numpy(), compression='lzf')
			
			# text
			with open(f'{root}/{name}/{id}.phn.txt', "r", encoding="utf8") as f:
				content = f.read()
				split = content.split(" ")
				phones = [f"<s>"] + [ " " if not p else p for p in split ] + [f"</s>"]
				for s in set(phones):
					if s not in symmap:
						symmap[s] = len(symmap.keys())
				phn = [ symmap[s] for s in phones ]

			group.create_dataset('text', data=phn, compression='lzf', chunks=True)

			# metadata
			group.attrs['id'] = id
			group.attrs['type'] = type
			group.attrs['speaker'] = name
			group.attrs['duration'] = qnt.shape[0] / 75
			group.attrs['phonemes'] = len(phn)

	# training
	for data_dir in tqdm(cfg.dataset.training, desc="Processing Training"):
		add( data_dir, type="training" )

	# validation
	for data_dir in tqdm(cfg.dataset.validation, desc='Processing Validation'):
		add( data_dir, type="validation" )

	# write symmap
	hf.create_dataset('symmap', data=json.dumps(symmap))

	hf.close()

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser("Save trained model to path.")
	parser.add_argument("--create-hdf5", action="store_true")
	args = parser.parse_args()

	if args.create_hdf5:
		create_dataset_hdf5()

	train_dl, subtrain_dl, val_dl = create_train_val_dataloader()
	print("Training DL:", next(iter(train_dl)))
	print("Training DL:", next(iter(train_dl)))
	print("Evaluation DL:", next(iter(subtrain_dl)))
	print("Evaluation DL:", next(iter(subtrain_dl)))
	print("Validation DL:", next(iter(val_dl)))
	print("Validation DL:", next(iter(val_dl)))
