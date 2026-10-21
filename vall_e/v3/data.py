# Dataset for the v3 prototype: NeMo-encoded utterances (.nem artifacts) with phonemes from the per-speaker metadata,
# plus a prompt cropped from another utterance of the same speaker.
import json
import math
import random
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from . import fsq

FPS = 44100 / 512  # NeMo 44KHz frames per second


def build_index(meta_root, data_root, groups):
	"""groups: [(relative dir, max speakers or None)]. Returns {"speakers": {key: [(path, phonemes)]}, "vocab": [chars]}."""
	speakers, chars = {}, set()
	for group, limit in groups:
		files = sorted((Path(meta_root) / group).glob("*.json"))[:limit]
		for f in files:
			items = []
			for uid, m in json.loads(f.read_text()).items():
				ph = m.get("phonemes")
				if ph:
					items.append((str(Path(data_root) / group / f.stem / f"{uid}.nem"), ph))
					chars.update(ph)
			if len(items) >= 2:
				speakers[f"{group}/{f.stem}"] = items
	return dict(speakers=speakers, vocab=sorted(chars))


def _frames(path):
	try:
		return int(np.load(path, allow_pickle=True)[()]["codes"].shape[-1])
	except (OSError, ValueError, KeyError):
		return -1


def add_frames(index, workers=8):
	"""Adds index["frames"][speaker] = [frames per utterance] (-1 if unreadable), for length bucketing. Reads every
	artifact once (about a minute per 500k utterances with 8 processes)."""
	paths = [(spk, path) for spk, items in index["speakers"].items() for path, _ in items]
	with Pool(workers) as pool:
		frames = pool.map(_frames, [p for _, p in paths], chunksize=256)
	out = {}
	for (spk, _), n in zip(paths, frames):
		out.setdefault(spk, []).append(n)
	index["frames"] = out
	return index


def is_val(speaker):
	return zlib.crc32(speaker.encode()) % 100 == 0


class Utterances(torch.utils.data.Dataset):
	def __init__(self, index, split, min_s=2.0, max_s=12.0, prompt_s=(3.0, 6.0), seed=0, crop_prompt=True):
		"""crop_prompt=False returns a window of the longest prompt length, for Collator to crop once per batch."""
		self.vocab = {c: i + 1 for i, c in enumerate(index["vocab"])}  # 0 is padding
		self.speakers = [s for s in index["speakers"] if is_val(s) == (split == "val")]
		self.items = [(s, i) for s in self.speakers for i in range(len(index["speakers"][s]))]
		self.index = index["speakers"]
		self.min_f, self.max_f = int(min_s * FPS), int(max_s * FPS)
		self.prompt_f = (int(prompt_s[0] * FPS), int(prompt_s[1] * FPS))
		self.seed = seed
		self.crop_prompt = crop_prompt
		frames = index.get("frames")
		self.frames = [frames[s][i] for s, i in self.items] if frames else None
		self.text_len = [len(self.index[s][i][1]) for s, i in self.items]

	def __len__(self):
		return len(self.items)

	def load(self, path):
		codes = np.load(path, allow_pickle=True)[()]["codes"]
		return fsq.codes_to_digits(torch.from_numpy(codes.astype(np.int64)))

	def tokenize(self, phonemes):
		return torch.tensor([self.vocab.get(c, 0) for c in phonemes if c in self.vocab], dtype=torch.long)

	def __getitem__(self, i):
		rng = random.Random(self.seed * 1_000_003 + i)
		for _ in range(20):  # skip missing files and out-of-range durations
			spk, j = self.items[i]
			path, ph = self.index[spk][j]
			try:
				target = self.load(path)
			except (OSError, ValueError, KeyError):
				target = None
			if target is not None and self.min_f <= len(target) <= self.max_f:
				break
			i = rng.randrange(len(self.items))
		else:
			raise RuntimeError(f"no loadable utterance in range after 20 tries (last: {path}); index paths are relative to the vall-e directory")
		others = [k for k in range(len(self.index[spk])) if k != j]
		prompt = None
		for k in rng.sample(others, min(5, len(others))):
			try:
				prompt = self.load(self.index[spk][k][0])
				break
			except (OSError, ValueError, KeyError):
				continue
		if prompt is None:
			prompt = target[:0]
		n = rng.randint(*self.prompt_f) if self.crop_prompt else self.prompt_f[1]
		if len(prompt) > n:
			s = rng.randrange(len(prompt) - n)
			prompt = prompt[s : s + n]
		return dict(text=self.tokenize(ph), prompt=prompt, target=target, speaker=spk, path=path)


def to_patches(digits, patch):
	"""[T, 32] -> [ceil(T / patch), patch * 32], padding by repeating the last frame."""
	pad = (-len(digits)) % patch
	if pad and len(digits):
		digits = torch.cat([digits, digits[-1:].expand(pad, -1)])
	return digits.reshape(-1, patch * fsq.SCALARS)


def collate(batch):
	return batch  # variable-length lists; the model packs them


def pack(text, prompt, target, masked, modes=None, pad_to=64):
	"""Lists per sample (text [Lt] ids, prompt [Lp, values] and target [La, values] digits, masked [La] bool, mode 0 for
	NAR-demask or 1 for AR) -> padded tensors for V3.hidden(). Each row is [text | prompt | target | padding], padded to
	a multiple of pad_to. kind: 0 padding, 1 text, 2 prompt, 3 target, 4 AR start.
	NAR rows: target positions hold the digits, `masked` ones replaced by the mask embedding in the model.
	AR rows: the target segment is [start, a_0 .. a_{n-1}], so position j predicts a_j from a_{j-1}, and n + 1 stop
	positions (the last one is the stop).
	tgt_index: flat indices into [B * L] of the positions predicting each target patch, in sample order (matching
	torch.cat(target)); tgt / tgt_masked: those patches and which are predicted (all of an AR row); hist / has_hist:
	each predicted patch's previous patch (AR only); stop_index / stop_tgt: the AR rows' stop positions and labels."""
	B = len(text)
	modes = modes or [0] * B
	lens = [len(t) + len(p) + len(a) + (m == 1) for t, p, a, m in zip(text, prompt, target, modes)]
	L = math.ceil(max(lens) / pad_to) * pad_to
	V, dev = target[0].shape[-1], target[0].device
	text_ids = torch.zeros(B, L, dtype=torch.long, device=dev)
	digits = torch.zeros(B, L, V, dtype=torch.int16, device=dev)
	kind = torch.zeros(B, L, dtype=torch.uint8, device=dev)
	pos_masked = torch.zeros(B, L, dtype=torch.bool, device=dev)
	tgt_index, tgt_masked, hist, has_hist, stop_index, stop_tgt = [], [], [], [], [], []
	for i, (t, p, a, m, mode) in enumerate(zip(text, prompt, target, masked, modes)):
		lt, lp, la = len(t), len(p), len(a)
		s0 = lt + lp
		text_ids[i, :lt] = t
		kind[i, :lt] = 1
		digits[i, lt:s0] = p
		kind[i, lt:s0] = 2
		if mode == 1:
			kind[i, s0] = 4
			digits[i, s0 + 1 : s0 + 1 + la] = a
			kind[i, s0 + 1 : s0 + 1 + la] = 3
			tgt_index.append(i * L + s0 + torch.arange(la, device=dev))
			tgt_masked.append(torch.ones(la, dtype=torch.bool, device=dev))
			hist.append(torch.cat([torch.zeros_like(a[:1]), a[:-1]]) if la else a)
			has_hist.append(torch.arange(la, device=dev) > 0)
			stop_index.append(i * L + s0 + torch.arange(la + 1, device=dev))
			stop_tgt.append(torch.arange(la + 1, device=dev) == la)
		else:
			digits[i, s0 : s0 + la] = a
			kind[i, s0 : s0 + la] = 3
			pos_masked[i, s0 : s0 + la] = m
			tgt_index.append(i * L + s0 + torch.arange(la, device=dev))
			tgt_masked.append(m)
			hist.append(torch.zeros_like(a))
			has_hist.append(torch.zeros(la, dtype=torch.bool, device=dev))
	cat = lambda xs, dtype: torch.cat(xs) if xs else torch.zeros(0, dtype=dtype, device=dev)
	return dict(text_ids=text_ids, digits=digits, kind=kind, masked=pos_masked, tgt_index=torch.cat(tgt_index),
		tgt=torch.cat([a.to(torch.int16) for a in target]), tgt_masked=torch.cat(tgt_masked),
		hist=torch.cat(hist).to(torch.int16), has_hist=torch.cat(has_hist),
		stop_index=cat(stop_index, torch.long), stop_tgt=cat(stop_tgt, torch.bool),
		mode=torch.tensor(modes, dtype=torch.long, device=dev),
		prefix_len=torch.tensor([len(t) + len(p) for t, p in zip(text, prompt)], dtype=torch.long, device=dev))


class Collator:
	"""Builds a training batch in the DataLoader worker: one prompt length per batch (3-6 s, cropped from each
	sample's window), a mask per target, both conditions dropped for cond_drop of samples, then pack().
	mask_ratio: None draws cos(u * pi / 2) per sample (covers every ratio the demasking loop visits); a float fixes it
	(AutoNAT, arXiv 2406.05478, found a fixed 0.8 on par with or slightly better than the cosine draw for images)."""
	def __init__(self, patch, prompt_f, cond_drop=0.1, mask_ratio=None, ar_frac=0.0):
		"""ar_frac: share of samples trained in AR mode (the rest NAR-demask)."""
		self.patch, self.prompt_f, self.cond_drop, self.mask_ratio, self.ar_frac = patch, prompt_f, cond_drop, mask_ratio, ar_frac

	def __call__(self, items):
		rng = random.Random()  # DataLoader seeds `random` per worker
		n = rng.randint(*self.prompt_f)
		text, prompt, target, masked, modes = [], [], [], [], []
		for it in items:
			p = it["prompt"]
			if len(p) > n:
				s = rng.randrange(len(p) - n)
				p = p[s : s + n]
			a = to_patches(it["target"], self.patch)
			r = math.cos(rng.random() * math.pi / 2) if self.mask_ratio is None else self.mask_ratio
			m = torch.zeros(len(a), dtype=torch.bool)
			m[torch.tensor(rng.sample(range(len(a)), max(1, round(r * len(a)))))] = True
			drop = rng.random() < self.cond_drop
			text.append(it["text"][:0] if drop else it["text"])
			prompt.append(to_patches(p, self.patch)[:0] if drop else to_patches(p, self.patch))
			target.append(a); masked.append(m)
			modes.append(1 if rng.random() < self.ar_frac else 0)
		return pack(text, prompt, target, masked, modes)


class BucketSampler(torch.utils.data.Sampler):
	"""Yields n_batches batches of similar total length: draws chunk * batch_size utterances at random, sorts them by
	length, cuts them into batches and shuffles the batches, so consecutive batches still differ in length.
	Utterances outside [min_f, max_f] frames (or unreadable, -1) are never drawn."""
	def __init__(self, ds, batch_size, n_batches, patch, chunk=64, seed=0):
		self.lengths = [tl + math.ceil(f / patch) for tl, f in zip(ds.text_len, ds.frames)]
		self.valid = [i for i, f in enumerate(ds.frames) if ds.min_f <= f <= ds.max_f]
		self.batch_size, self.n_batches, self.chunk, self.seed = batch_size, n_batches, chunk, seed

	def __len__(self):
		return self.n_batches

	def __iter__(self):
		rng = random.Random(self.seed)
		done = 0
		while done < self.n_batches:
			idx = sorted((rng.choice(self.valid) for _ in range(self.chunk * self.batch_size)), key=self.lengths.__getitem__)
			batches = [idx[i : i + self.batch_size] for i in range(0, len(idx), self.batch_size)]
			rng.shuffle(batches)
			for b in batches[: self.n_batches - done]:
				yield b
			done += len(batches)


class StoreUtterances(torch.utils.data.Dataset):
	"""Utterances from a v3 Store (vall_e/v3/store.py), with the same interface as Utterances: items, frames, text_len,
	prompt_f, and items of dict(text, prompt, target, speaker, path, text_raw). Speakers split train/val by is_val(),
	so a speaker never appears in both. groups / languages filter by name (None: all)."""
	def __init__(self, store, split, min_s=2.0, max_s=12.0, prompt_s=(3.0, 6.0), seed=0, crop_prompt=True, groups=None, languages=None):
		from .store import Store
		self.store = store if isinstance(store, Store) else Store(store)
		st = self.store
		keys = [f"{st.m['groups'][g]}/{s}" for s, g in zip(st.m["speakers"], st.m["speaker_group"])]
		spk_ok = np.array([is_val(k) == (split == "val") for k in keys])
		ok = spk_ok[st.speaker]
		if groups:
			gids = [i for i, g in enumerate(st.m["groups"]) if any(g == x or g.startswith(x + "/") for x in groups)]
			ok &= np.isin(st.group, gids)
		if languages:
			ok &= np.isin(st.language, [st.m["languages"].index(l) for l in languages if l in st.m["languages"]])
		self.items = np.nonzero(ok)[0]
		self.frames = st.frames[self.items].tolist()
		phon_len = np.concatenate([np.diff(np.asarray(sh["phoneme_offsets"])) for sh in st.shards])
		self.text_len = phon_len[self.items].tolist()
		self.vocab_size = len(st.m["vocab"]) + 1  # ids + 1; 0 is padding
		self.min_f, self.max_f = int(min_s * FPS), int(max_s * FPS)
		self.prompt_f = (int(prompt_s[0] * FPS), int(prompt_s[1] * FPS))
		self.seed, self.crop_prompt = seed, crop_prompt

	def __len__(self):
		return len(self.items)

	def __getitem__(self, i):
		st = self.store
		rng = random.Random(self.seed * 1_000_003 + i)
		u = int(self.items[i])
		for _ in range(20):  # out-of-range durations (the bucketed sampler already avoids them)
			if self.min_f <= st.frames[u] <= self.max_f:
				break
			u = int(self.items[rng.randrange(len(self.items))])
		target = fsq.codes_to_digits(torch.from_numpy(st.codes(u).astype(np.int64)).t())
		others = st.utterances_of(int(st.speaker[u]))
		others = others[others != u]
		prompt = target[:0]
		if len(others):
			k = int(others[rng.randrange(len(others))])
			prompt = fsq.codes_to_digits(torch.from_numpy(st.codes(k).astype(np.int64)).t())
		n = rng.randint(*self.prompt_f) if self.crop_prompt else self.prompt_f[1]
		if len(prompt) > n:
			s = rng.randrange(len(prompt) - n)
			prompt = prompt[s : s + n]
		text = torch.from_numpy(st.phonemes(u).astype(np.int64) + 1)
		return dict(text=text, prompt=prompt, target=target, speaker=st.speaker_key(u), path=f"{st.root}#{u}", text_raw=st.text(u))
