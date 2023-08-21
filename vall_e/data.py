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
from .emb.qnt import trim_random, repeat_extend_audio, merge_audio, decode_to_file

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

def get_task_symmap():
	start = 1024
	symmap = {
		"<tts>": -100,
		"<ns>": start + 0,
		"<sr>": start + 1,
		"<tse>": start + 2,
		"<soe>": start + 3,
		"<mask>": start + 4,
		"<eoe>": start + 5,
		"<svc>": start + 6,
	}
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
	return torch.load(path)[0][:, :].t().to(torch.int16)

@cache
def _get_phones(path, language="en"):
	content = open(_get_phone_path(path), "r", encoding="utf8").read().split(" ")
	return ["<s>"] + [ " " if not p else p for p in split ] + ["</s>"]

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
		task_symmap=None,
		min_phones=cfg.dataset.phones_range[0],
		max_phones=cfg.dataset.phones_range[1],
		min_duration=cfg.dataset.duration_range[0],
		max_duration=cfg.dataset.duration_range[1],
		training=False,
		extra_paths_by_spkr_name: dict[str, list] = {},
	):
		super().__init__()
		self._head = None
		self.min_phones = min_phones
		self.max_phones = max_phones
		self.min_duration = min_duration
		self.max_duration = max_duration
		self.sampler = None

		if cfg.dataset.validate:
			self.paths = [
				path for path in paths if _validate(path, self.min_phones, self.max_phones, self.min_duration, self.max_duration)
			]
		else:
			self.paths = paths

		self.phone_symmap = phone_symmap or self._get_phone_symmap()
		self.spkr_symmap = spkr_symmap or self._get_spkr_symmap()
		self.task_symmap = get_task_symmap or self._get_task_symmap()
		self.training = training

		# assert len(self.phone_symmap) < 256, "Unique token count should be [0,255] to fit within uint8"
		self.text_dtype = torch.uint8 if len(self.phone_symmap) < 256 else torch.int16

		self.paths_by_spkr_name = self._get_paths_by_spkr_name(extra_paths_by_spkr_name)

		if cfg.dataset.validate:
			self.paths = [
				p for p in self.paths if len(self.paths_by_spkr_name[cfg.get_spkr(p)]) > 1
			]

		if cfg.dataset.sample_type == "path":
			self.paths = [*_interleaved_reorder(self.paths, cfg.get_spkr)]

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

	@cached_property
	def spkrs(self):
		return sorted({cfg.get_spkr(path) for path in self.paths})

	@cached_property
	def tasks(self):
		return cfg.dataset.tasks_list # ["tts", "tts", "ns", "sr", "tse", "tts", "tts"] # , "cse", "nse"

	def _get_phone_symmap(self):
		return get_phone_symmap()

	def _get_spkr_symmap(self):
		return {s: i for i, s in enumerate(self.spkrs)}

	def _get_task_symmap(self):
		return get_task_symmap()

	def get_task_token( self, token, levels=cfg.models.max_levels ):
		if not hasattr(self, "task_symmap"):
			self.task_symmap = self._get_task_symmap()
		return torch.Tensor([[ self.task_symmap[f'<{token}>'] for _ in range(levels) ]]).to(dtype=torch.int16)

	def sample_noise(self):
		paths = []
		for data_dir in cfg.dataset.noise:
			paths.extend(data_dir.rglob("*.qnt.pt"))
		path = random.choice(paths)

		if False and cfg.dataset.use_hdf5:
			key = f'/noise/{_get_hdf5_path(path)}'
			qnt = torch.from_numpy(cfg.hdf5[key]["audio"][:, :]).to(torch.int16)
		else:
			qnt = _load_quants(path)
		return qnt

	def sample_speakers(self, ignore=[]):
		choices = set(self.spkrs) - set(ignore)
		return random.choice([*choices])

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
		prom_length = 0
		trim_length = int(cfg.dataset.prompt_duration * 75) + random.randint(-16, 16)

		for _ in range(cfg.dataset.max_prompts):
			path = random.choice(choices)
			if cfg.dataset.use_hdf5:
				key = _get_hdf5_path(path)
				qnt = torch.from_numpy(cfg.hdf5[key]["audio"][:, :]).to(torch.int16)
			else:
				qnt = _load_quants(path)

			if cfg.dataset.prompt_duration > 0 and trim_length < qnt.shape[0]:
				qnt = trim_random( qnt, trim_length )

			prom_list.append(qnt)
			prom_length += qnt.shape[0]

			if prom_length >= trim_length or random.random() > cfg.dataset.random_utterance:
				break

		prom = torch.cat(prom_list)

		if cfg.dataset.prompt_duration > 0 and trim_length < prom.shape[0]:
			prom = trim_random( prom, trim_length )

		return prom

	def __getitem__(self, index):
		if cfg.dataset.sample_type == "speaker":
			spkr_name = self.spkrs[index]
			spkr_id = self.spkr_symmap[spkr_name]
			path = random.choice([*set(self.paths_by_spkr_name[spkr_name])])
		else:
			path = self.paths[index]
			spkr_name = cfg.get_spkr(path)
			spkr_id = self.spkr_symmap[spkr_name]

		if cfg.dataset.use_hdf5:
			key = _get_hdf5_path(path)
			text = torch.from_numpy(cfg.hdf5[key]["text"][:]).to(self.text_dtype)
			resps = torch.from_numpy(cfg.hdf5[key]["audio"][:, :]).to(torch.int16)
		else:
			text = torch.tensor([*map(self.phone_symmap.get, _get_phones(path))]).to(self.text_dtype)
			resps = _load_quants(path)
		
		task = random.choice(self.tasks)

		# ensure a speaker has at least four utterances
		# default to tts if not
		if len(set(self.paths_by_spkr_name[spkr_name]) - {path}) < 4:
			task = "tts"

		noise_scale = 0.25
		# text-to-speech
		if task == "tts":
			proms = self.sample_prompts(spkr_name, ignore=path) if random.random() < cfg.dataset.random_utterance else resps
		# noise suppression || speech removal
		elif task == "ns" or task == "sr":
			# sample random noise
			noise = self.sample_noise()
			# extend the noise to fill the target audio
			noise = repeat_extend_audio(noise, resps.shape[0])
			# create the input prompt by merging the target audio with the noise
			proms = merge_audio( resps, noise, scale=[1, noise_scale], device="cpu" )
			# set the target to just be the noise if <sr>
			if task == "sr":
				resps = noise
			# prepend the task token
			proms = torch.cat( [self.get_task_token(task), proms] )

			# set the text prompt to empty to train without a guided text prompt
			if random.random() < 0.5:
				text = torch.tensor([1, 2]).to(self.text_dtype)
		# target speech extraction
		elif task == "tse":
			# sample a random, clean, utterance for the target speaker
			clean_proms = self.sample_prompts(spkr_name, ignore=path)
			# sample a random, clean utterance from a different speaker
			other_proms = self.sample_prompts(self.sample_speakers(ignore=[spkr_name]), ignore="")
			# overlay the random speaker over the target audio

			smallest_size = min(resps.shape[0], other_proms.shape[0])
			if other_proms.shape[0] == smallest_size:
				noisy_proms = merge_audio( resps[:smallest_size, :], other_proms, scale=[1, random.uniform(0.5, 0.75)], device="cpu" )
				noisy_proms = torch.cat( [ noisy_proms, resps[smallest_size:, :] ] )
			else:
				noisy_proms = merge_audio( resps, other_proms[:smallest_size, :], scale=[1, random.uniform(0.5, 0.75)], device="cpu" )
				noisy_proms = torch.cat( [ noisy_proms, other_proms[smallest_size:, :] ] )

			# stitch together the promps			
			proms = torch.cat( [clean_proms, self.get_task_token(task), noisy_proms] )

			# set the text prompt to empty to train without a guided text prompt
			if random.random() < 0.5:
				text = torch.tensor([1, 2]).to(self.text_dtype) # <s></s>

		# speech editing would require higher quality transcription data (phoneme level/word level) unfortunately
		# as I need to get a good clean point to trim into
		# clean speech editing
		elif task == "cse" or task == "nse":
			choices = set(self.paths_by_spkr_name[spkr_name]) - {path}
			sampled = random.sample([*choices], 4)

			if cfg.dataset.use_hdf5:
				texts = [ torch.from_numpy(cfg.hdf5[_get_hdf5_path(path)]["text"][:]).to(self.text_dtype) for path in sampled ]
				qnts = [ torch.from_numpy(cfg.hdf5[_get_hdf5_path(path)]["audio"][:, :]).to(torch.int16) for path in sampled ]
			else:
				texts = [ torch.tensor([*map(self.phone_symmap.get, _get_phones(path))]).to(self.text_dtype) for path in sampled ]
				qnts = [ _load_quants(path) for path in sampled ]

			# remove <s></s>
			for i in range(len(texts)):
				texts[i] = texts[i][1:-1]

			pre_text, mid_text, post_text, edit_text = texts
			pre_prom, mid_prom, post_prom, edit_prom = qnts

			# randomly drop out pre
			if random.random() < 0.125:
				pre_text = None
				pre_prom = None
			# randomly drop out post
			if random.random() < 0.125:
				post_text = None
				post_prom = None

			# create new text
			text = torch.cat(
				[ torch.Tensor( [ 1 ] ).to(dtype=self.text_dtype) ] + # <s>
				([ pre_text, torch.Tensor( [ 3 ] ).to(dtype=self.text_dtype) ] if pre_text is not None else []) + # pre_text + space'
				[ edit_text ] + # 'edit text'
				([ torch.Tensor( [ 3 ] ).to(dtype=self.text_dtype), post_text ] if post_text is not None else []) + # 'space' + edit_text
				[ torch.Tensor( [ 2 ] ).to(dtype=self.text_dtype) ] # </s>
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
					return merge_audio(p, n, scale=[1, noise_scale], device="cpu")
				
				# apply noise to all pieces
				pre_prom = noise_proms( pre_prom )
				mid_prom = noise_proms( mid_prom )
				post_prom = noise_proms( post_prom )
				edit_prom = noise_proms( edit_prom )
			else:
				mid_prom = self.get_task_token("mask")

			# create new proms
			proms = torch.cat( 
				([ pre_prom ] if pre_prom is not None else []) +
				[self.get_task_token("soe")] +
				[ mid_prom ] + # is <mask> if task is CSE
				[self.get_task_token("eoe")] +
				([ post_prom ] if post_prom is not None else [])
			)
			# create new resp
			resps = torch.cat( 
				([ pre_prom ] if pre_prom is not None else []) +
				[ edit_prom ] +
				([ post_prom ] if post_prom is not None else [])
			)
		else:
			raise f'Undefined task: {task}'

		"""
		# emulate SVC
		# takes in an utterance of the target speaker, a target utterenace as a reference clip as the input prompt
		# targets an utterance of the target speaker with the same tempo + pitch + etc as the reference clip

		# NOTE: I do not have a clue how to go about this. I *could* dynamically generate clips through RVC here, but I imagine the penalty would be astronomical
		# ahead-of-time dataset preparation of a shit ton of RVC clips might be the key.
		# aside from that, I have no clue how to go about training this, as this is entirely a proof of concept task.
		elif task == "svc":
			# sample a random, clean utterance for the target speaker
			proms = self.sample_prompts(spkr_name, ignore=path) if random.random() < cfg.dataset.random_utterance else resps
			# sample a reference clip from a different speaker
			ref_proms = self.sample_rvc(self.sample_speakers(ignore=[spkr_name]))
			# 
			resps = 
			# stitch together the promps
			proms = torch.cat( [proms, self.get_task_token(task), ref_proms] )

			# set the text prompt to empty to train without a guided text prompt
			if random.random() < 0.5:
				text = torch.tensor([1, 2]).to(self.text_dtype)
		"""

		# trim to fit to requested prom/resps levels
		proms = proms[:, :cfg.models.prom_levels]
		resps = resps[:, :cfg.models.prom_levels]


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

	def __len__(self):
		if cfg.dataset.sample_type == "speaker":
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

	val_dataset.head_(cfg.evaluation.size)

	return train_dataset, val_dataset


def create_train_val_dataloader():
	train_dataset, val_dataset = create_datasets()

	subtrain_dataset = copy.deepcopy(train_dataset)
	if cfg.dataset.sample_type == "path":
		subtrain_dataset.head_(cfg.evaluation.size)

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

# parse yaml to create an hdf5 file
def create_dataset_hdf5():
	cfg.dataset.use_hdf5 = True
	cfg.load_hdf5(write=True)

	symmap = get_phone_symmap()
	
	root = cfg.cfg_path
	hf = cfg.hdf5

	def add( dir, type="training", audios=True, texts=True ):
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
			audio_exists = os.path.exists(f'{root}/{name}/{id}.qnt.pt') if audios else True
			text_exists = os.path.exists(f'{root}/{name}/{id}.phn.txt') if texts else True

			if not audio_exists or not text_exists:
				continue

			key = f'{type}/{name}/{id}'
			if key in hf:
				# print("Skipping existing entry:", key)
				continue

			group = hf.create_group(key)

			# audio
			if audios:
				qnt = torch.load(f'{root}/{name}/{id}.qnt.pt')[0].t()
				group.create_dataset('audio', data=qnt.numpy(), compression='lzf')
			
			# text
			if texts:
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

	# noise
	for data_dir in tqdm(cfg.dataset.noise, desc='Processing Noise'):
		add( data_dir, type="noise", texts=False )

	# write symmap
	try:
		hf.create_dataset('symmap', data=json.dumps(symmap))
	except Exception as e:
		pass

	hf.close()

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser("Save trained model to path.")
	parser.add_argument("--action", type=str)
	parser.add_argument("--tasks", type=str)
	args = parser.parse_args()

	task = args.action

	cfg.dataset.workers = 1

	if args.action == "hdf5":
		create_dataset_hdf5()
	elif args.action == "sample":
		train_dl, subtrain_dl, val_dl = create_train_val_dataloader()

		samples = {
			"training": [ next(iter(train_dl)),  next(iter(train_dl)) ],
			"evaluation": [ next(iter(subtrain_dl)),  next(iter(subtrain_dl)) ],
			"validation": [ next(iter(val_dl)),  next(iter(val_dl)) ],
		}

		for k, v in samples.items():
			for i in range(len(v)):
				del v[i]['proms']
				del v[i]['resps']
			print(f'{k}:', v)
	elif args.action == "tasks":
		index = 0
		cfg.dataset.tasks_list = args.tasks.split(",")
		
		train_dl, subtrain_dl, val_dl = create_train_val_dataloader()
		batch = next(iter(train_dl))

		for text, resps, proms, task in zip(batch["text"], batch["resps"], batch["proms"], batch["task"]):
			if task not in cfg.dataset.tasks_list:
				continue

			print(text, task, cfg.models.prom_levels)
			print( proms.shape, resps.shape )
			decode_to_file( proms, f"./data/{task}.proms.wav", device="cpu" )
			decode_to_file( resps, f"./data/{task}.resps.wav", device="cpu" )
			break