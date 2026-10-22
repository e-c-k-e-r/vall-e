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


def pack(text, prompt, target, masked, modes=None, pad_to=64, langs=None):
	"""Lists per sample (text [Lt] ids, prompt [Lp, values] and target [La, values] digits, masked [La] bool, mode 0 for
	NAR-demask, 1 for AR or 2 for LEN; langs: language ids, 0 unknown) -> padded tensors for V3.hidden(). Each row is
	[text | prompt | target | padding], padded to a multiple of pad_to. kind: 0 padding, 1 text, 2 prompt, 3 target,
	4 AR start, 5 length query.
	LEN rows are [text | prompt | query]: the query predicts the target's length in patches (len_index / len_tgt); their
	targets are not predicted, so they're left out of tgt.
	NAR rows: target positions hold the digits, `masked` ones replaced by the mask embedding in the model.
	AR rows: the target segment is [start, a_0 .. a_{n-1}], so position j predicts a_j from a_{j-1}, and n + 1 stop
	positions (the last one is the stop).
	tgt_index: flat indices into [B * L] of the positions predicting each target patch, in sample order (matching
	torch.cat(target)); tgt / tgt_masked: those patches and which are predicted (all of an AR row); hist / has_hist:
	each predicted patch's previous patch (AR only); stop_index / stop_tgt: the AR rows' stop positions and labels."""
	B = len(text)
	modes = modes or [0] * B
	lens = [len(t) + len(p) + (1 if m == 2 else len(a) + (m == 1)) for t, p, a, m in zip(text, prompt, target, modes)]
	L = math.ceil(max(lens) / pad_to) * pad_to
	V, dev = target[0].shape[-1], target[0].device
	text_ids = torch.zeros(B, L, dtype=torch.long, device=dev)
	digits = torch.zeros(B, L, V, dtype=torch.int16, device=dev)
	kind = torch.zeros(B, L, dtype=torch.uint8, device=dev)
	pos_masked = torch.zeros(B, L, dtype=torch.bool, device=dev)
	tgt_index, tgt_masked, hist, has_hist, stop_index, stop_tgt, tgts, len_index, len_tgt = [], [], [], [], [], [], [], [], []
	for i, (t, p, a, m, mode) in enumerate(zip(text, prompt, target, masked, modes)):
		lt, lp, la = len(t), len(p), len(a)
		s0 = lt + lp
		text_ids[i, :lt] = t
		kind[i, :lt] = 1
		digits[i, lt:s0] = p
		kind[i, lt:s0] = 2
		if mode == 2:
			kind[i, s0] = 5
			len_index.append(i * L + s0); len_tgt.append(la)
			continue
		tgts.append(a)
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
	return dict(text_ids=text_ids, digits=digits, kind=kind, masked=pos_masked, tgt_index=cat(tgt_index, torch.long),
		tgt=cat([a.to(torch.int16) for a in tgts], torch.int16).view(-1, V), tgt_masked=cat(tgt_masked, torch.bool),
		hist=cat(hist, torch.int16).to(torch.int16).view(-1, V), has_hist=cat(has_hist, torch.bool),
		len_index=torch.tensor(len_index, dtype=torch.long, device=dev), len_tgt=torch.tensor(len_tgt, dtype=torch.float, device=dev),
		lang=torch.tensor(langs if langs is not None else [0] * B, dtype=torch.long, device=dev),
		stop_index=cat(stop_index, torch.long), stop_tgt=cat(stop_tgt, torch.bool),
		mode=torch.tensor(modes, dtype=torch.long, device=dev),
		prefix_len=torch.tensor([len(t) + len(p) for t, p in zip(text, prompt)], dtype=torch.long, device=dev))


class Collator:
	"""Builds a training batch in the DataLoader worker: one prompt length per batch (cropped from each sample's
	window), a mask per target, the text and the prompt each dropped independently (text_drop / prompt_drop: both for
	CFG's null, prompt alone for promptless generation), the language id dropped for lang_drop, then pack().
	mask_ratio: None draws cos(u * pi / 2) per sample (covers every ratio the demasking loop visits); a float fixes it
	(AutoNAT, arXiv 2406.05478, found a fixed 0.8 on par with or slightly better than the cosine draw for images)."""
	def __init__(self, patch, prompt_f, text_drop=0.1, prompt_drop=0.1, mask_ratio=0.8, ar_frac=0.0, len_frac=0.0, lang_drop=0.2):
		"""ar_frac / len_frac: shares of samples trained in AR and LEN mode (the rest NAR-demask)."""
		self.patch, self.prompt_f, self.mask_ratio, self.ar_frac, self.len_frac = patch, prompt_f, mask_ratio, ar_frac, len_frac
		self.text_drop, self.prompt_drop, self.lang_drop = text_drop, prompt_drop, lang_drop

	def __call__(self, items):
		rng = random.Random()  # DataLoader seeds `random` per worker
		n = rng.randint(*self.prompt_f)
		text, prompt, target, masked, modes, langs = [], [], [], [], [], []
		for it in items:
			p = it["prompt"]
			if len(p) > n:
				s = rng.randrange(len(p) - n)
				p = p[s : s + n]
			a = to_patches(it["target"], self.patch)
			r = math.cos(rng.random() * math.pi / 2) if self.mask_ratio is None else self.mask_ratio
			m = torch.zeros(len(a), dtype=torch.bool)
			m[torch.tensor(rng.sample(range(len(a)), max(1, round(r * len(a)))))] = True
			text.append(it["text"][:0] if rng.random() < self.text_drop else it["text"])
			prompt.append(to_patches(p, self.patch)[:0] if rng.random() < self.prompt_drop else to_patches(p, self.patch))
			target.append(a); masked.append(m)
			u = rng.random()
			modes.append(1 if u < self.ar_frac else 2 if u < self.ar_frac + self.len_frac else 0)
			langs.append(0 if rng.random() < self.lang_drop else it.get("lang", 0))
		return pack(text, prompt, target, masked, modes, langs=langs)


class BucketSampler(torch.utils.data.Sampler):
	"""Yields n_batches batches of similar total length: draws chunk * batch_size utterances at random, sorts them by
	length, cuts them into batches and shuffles the batches, so consecutive batches still differ in length.
	Utterances outside [min_f, max_f] frames (or unreadable, -1) are never drawn. weights (per item, optional): draw
	probabilities, e.g. Dataset.speaker_weights() to even out speakers. max_tokens (optional): batches whose rows x
	(longest row + extra) exceed it are split, extra being positions the collator adds (the longest prompt), so the
	longest batches fit in memory without shrinking the typical one."""
	def __init__(self, ds, batch_size, n_batches, patch, chunk=64, seed=0, weights=None, max_tokens=None, extra=0):
		frames = np.asarray(ds.frames)
		self.lengths = np.asarray(ds.text_len) + -(-frames // patch)
		self.valid = np.nonzero((frames >= ds.min_f) & (frames <= ds.max_f))[0]
		self.cum = None
		if weights is not None:
			w = np.asarray(weights, dtype=np.float64)[self.valid]
			self.cum = np.cumsum(w / w.sum())
		self.batch_size, self.n_batches, self.chunk, self.seed = batch_size, n_batches, chunk, seed
		self.max_tokens, self.extra = max_tokens, extra

	def __len__(self):
		return self.n_batches

	def __iter__(self):
		rng = random.Random(self.seed)
		done = 0
		while done < self.n_batches:
			nrng = np.random.default_rng(rng.randrange(2**32))
			n = self.chunk * self.batch_size
			pick = nrng.integers(len(self.valid), size=n) if self.cum is None else np.minimum(np.searchsorted(self.cum, nrng.random(n)), len(self.valid) - 1)
			idx = self.valid[pick]
			idx = idx[np.argsort(self.lengths[idx], kind="stable")].tolist()
			batches = [idx[i : i + self.batch_size] for i in range(0, len(idx), self.batch_size)]
			if self.max_tokens:
				split = []
				for b in batches:
					rows = max(1, self.max_tokens // (int(self.lengths[b[-1]]) + self.extra))  # sorted: b[-1] is the longest
					# into equal parts (16 rows at 15 per batch is 8 + 8, not 15 + 1)
					size = -(-len(b) // -(-len(b) // rows))
					split += [b[i : i + size] for i in range(0, len(b), size)]
				batches = split
			rng.shuffle(batches)
			for b in batches[: self.n_batches - done]:
				yield b
			done += len(batches)


class StoreUtterances(torch.utils.data.Dataset):
	"""Utterances from a v3 Store (vall_e/v3/store.py), with the same interface as Utterances: items, frames, text_len,
	prompt_f, and items of dict(text, prompt, target, speaker, path, text_raw, lang). Speakers split train/val by
	is_val(), so a speaker never appears in both. groups / languages filter by name (None: all); the store's exclude
	list is always applied.
	Prompts: with similar_p, one of the target's similar_k most similar utterances (the store's similar lists), else a
	random one of the speaker's; with noisy_p, the speaker's noised prompt instead (if the store has one, from another
	utterance). lang is the store's language id + 1 (0 is unknown)."""
	def __init__(self, store, split, min_s=2.0, max_s=12.0, prompt_s=(3.0, 6.0), seed=0, crop_prompt=True, groups=None, languages=None,
			similar_p=0.0, similar_k=3, noisy_p=0.0):
		from .store import Store
		self.store = store if isinstance(store, Store) else Store(store)
		st = self.store
		keys = [f"{st.m['groups'][g]}/{s}" for s, g in zip(st.m["speakers"], st.m["speaker_group"])]
		spk_ok = np.array([is_val(k) == (split == "val") for k in keys])
		ok = spk_ok[st.speaker]
		if groups:
			unknown = [x for x in groups if not any(g == x or g.startswith(x + "/") for g in st.m["groups"])]
			if unknown:
				raise ValueError(f"no store group matches {unknown}; groups are {st.m['groups']}")
			gids = [i for i, g in enumerate(st.m["groups"]) if any(g == x or g.startswith(x + "/") for x in groups)]
			ok &= np.isin(st.group, gids)
		if languages:
			ok &= np.isin(st.language, [st.m["languages"].index(l) for l in languages if l in st.m["languages"]])
		if st.exclude is not None:
			ok &= ~np.asarray(st.exclude)
		self.items = np.nonzero(ok)[0]
		self.frames = st.frames[self.items].tolist()
		phon_len = np.concatenate([np.diff(np.asarray(sh["phoneme_offsets"])) for sh in st.shards])
		self.text_len = phon_len[self.items].tolist()
		self.vocab_size = len(st.m["vocab"]) + 1  # ids + 1; 0 is padding
		self.min_f, self.max_f = int(min_s * FPS), int(max_s * FPS)
		self.prompt_f = (int(prompt_s[0] * FPS), int(prompt_s[1] * FPS))
		self.seed, self.crop_prompt = seed, crop_prompt
		self.similar_p, self.similar_k, self.noisy_p = similar_p, similar_k, noisy_p
		self.n_lang = len(st.m["languages"]) + 1

	def speaker_weights(self, alpha=0.5):
		"""Per-item draw weights n_speaker ** -alpha: 0 keeps the utterance distribution, 1 draws every speaker equally."""
		spk = self.store.speaker[self.items]
		return np.bincount(spk)[spk].astype(np.float64) ** -alpha

	def _prompt_source(self, u, rng):
		"""The utterance (or noised prompt, as ("noisy", k)) to crop the prompt from, or None."""
		st = self.store
		spk = int(st.speaker[u])
		if self.noisy_p and st.noisy is not None and rng.random() < self.noisy_p:
			k = int(st.noisy_of[spk])
			if k >= 0 and int(st.noisy_source[k]) != u:
				return ("noisy", k)
		if st.similar is not None and rng.random() < self.similar_p:
			sim = [int(v) for v in st.similar[u][: self.similar_k] if v >= 0 and not (st.exclude is not None and st.exclude[v])]
			if sim:
				return sim[rng.randrange(len(sim))]
		others = st.utterances_of(spk)
		others = others[others != u]
		return int(others[rng.randrange(len(others))]) if len(others) else None

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
		src = self._prompt_source(u, rng)
		prompt = target[:0]
		if src is not None:
			codes = st.noisy_codes(src[1]) if isinstance(src, tuple) else st.codes(src)
			prompt = fsq.codes_to_digits(torch.from_numpy(codes.astype(np.int64)).t())
		n = rng.randint(*self.prompt_f) if self.crop_prompt else self.prompt_f[1]
		if len(prompt) > n:
			s = rng.randrange(len(prompt) - n)
			prompt = prompt[s : s + n]
		text = torch.from_numpy(st.phonemes(u).astype(np.int64) + 1)
		return dict(text=text, prompt=prompt, target=target, speaker=st.speaker_key(u), path=f"{st.root}#{u}", text_raw=st.text(u),
			lang=int(st.language[u]) + 1)
