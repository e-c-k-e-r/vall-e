# The v3 dataset store: one canonical copy of the encoded audio and its index, memory-mapped for training.
#
#   <root>/manifest.json         codec, frame rate, shard list, and the tables the per-utterance ids index into:
#                                groups, speakers (with their group), languages, phoneme vocabulary
#   <root>/<shard>/codes.u16     every utterance's codes concatenated, raw uint16 [frames, 8], append-only
#   <root>/<shard>/*.npy         per-utterance arrays: frame_offsets, speaker, language, phoneme ids and offsets,
#                                UTF-8 text and offsets, UTF-8 names and offsets
#
# New data is new shards; nothing is rewritten. Readers keep only small global arrays in memory (frames, speaker,
# language) and memory-map the rest, so DataLoader workers share pages instead of copying the index.
#
#   python -m vall_e.v3.store convert --meta training/llama-nemo/metadata --data training/llama-nemo/data --out <root>
import argparse
import json
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

CODEC = dict(name="nvidia/audio-codec-44khz", sample_rate=44100, samples_per_frame=512, groups=8, levels=[8, 5, 5, 5])


class ShardWriter:
	def __init__(self, path):
		self.path = Path(path)
		self.path.mkdir(parents=True, exist_ok=True)
		self.codes = open(self.path / "codes.u16", "wb")
		self.frames = [0]
		self.speaker, self.language = [], []
		self.phon, self.phon_off = [], [0]
		self.text, self.text_off = bytearray(), [0]
		self.name, self.name_off = bytearray(), [0]

	@property
	def n_frames(self):
		return self.frames[-1]

	def add(self, codes, speaker, language, phon_ids, text, name):
		"""codes [8, T] integer codes, as the .nem artifacts store them (leading singleton dims, as some artifacts have
		[1, 8, T], are dropped); phon_ids a uint16 array of ids."""
		codes = np.asarray(codes)
		while codes.ndim > 2 and codes.shape[0] == 1:
			codes = codes[0]
		if codes.ndim != 2 or codes.shape[0] != CODEC["groups"]:
			raise ValueError(f"{name}: expected codes [{CODEC['groups']}, T], got {codes.shape}")
		codes = np.ascontiguousarray(codes.T, dtype=np.uint16)  # stored [T, 8]
		self.codes.write(codes.tobytes())
		self.frames.append(self.frames[-1] + len(codes))
		self.speaker.append(speaker); self.language.append(language)
		self.phon.append(np.asarray(phon_ids, dtype=np.uint16)); self.phon_off.append(self.phon_off[-1] + len(phon_ids))
		self.text += text.encode(); self.text_off.append(len(self.text))
		self.name += name.encode(); self.name_off.append(len(self.name))

	def close(self):
		self.codes.close()
		size = (self.path / "codes.u16").stat().st_size
		if size != self.n_frames * CODEC["groups"] * 2:
			raise RuntimeError(f"{self.path}: codes file holds {size} bytes, the index accounts for {self.n_frames} frames")
		save = lambda k, v, dt: np.save(self.path / f"{k}.npy", np.asarray(v, dtype=dt))
		save("frame_offsets", self.frames, np.int64)
		save("speaker", self.speaker, np.int32)
		save("language", self.language, np.uint8)
		save("phonemes", np.concatenate(self.phon) if self.phon else [], np.uint16)
		save("phoneme_offsets", self.phon_off, np.int64)
		save("text", np.frombuffer(bytes(self.text), dtype=np.uint8), np.uint8)
		save("text_offsets", self.text_off, np.int64)
		save("names", np.frombuffer(bytes(self.name), dtype=np.uint8), np.uint8)
		save("name_offsets", self.name_off, np.int64)
		return dict(name=self.path.name, utterances=len(self.speaker), frames=self.n_frames)


class StoreWriter:
	"""Appends utterances to <root>, starting a new shard every shard_frames frames. Reopening an existing store
	continues its tables and adds shards after the existing ones."""
	def __init__(self, root, shard_frames=128_000_000):
		self.root = Path(root)
		self.root.mkdir(parents=True, exist_ok=True)
		mpath = self.root / "manifest.json"
		self.m = json.loads(mpath.read_text()) if mpath.exists() else dict(codec=CODEC, shards=[], groups=[], speakers=[],
			speaker_group=[], languages=[], vocab=[])
		self.shard_frames = shard_frames
		self._ids = {k: {v: i for i, v in enumerate(self.m[k])} for k in ("groups", "languages", "vocab")}
		self._spk = {(self.m["speaker_group"][i], s): i for i, s in enumerate(self.m["speakers"])}
		self.shard = None
		# code point -> phoneme id, so a whole string maps with one array lookup
		self._lut = np.full(0x110000, -1, dtype=np.int32)
		for i, c in enumerate(self.m["vocab"]):
			self._lut[ord(c)] = i

	def _id(self, table, value):
		ids = self._ids[table]
		if value not in ids:
			ids[value] = len(self.m[table]); self.m[table].append(value)
		return ids[value]

	def speaker_id(self, group, speaker):
		g = self._id("groups", group)
		if (g, speaker) not in self._spk:
			self._spk[(g, speaker)] = len(self.m["speakers"])
			self.m["speakers"].append(speaker); self.m["speaker_group"].append(g)
		return self._spk[(g, speaker)]

	def add(self, codes, group, speaker, language, phonemes, text, name):
		if self.shard is None or self.shard.n_frames >= self.shard_frames:
			self._close_shard()
			self.shard = ShardWriter(self.root / f"shard-{len(self.m['shards']):05d}")
		cps = np.frombuffer(phonemes.encode("utf-32-le"), dtype=np.uint32) if isinstance(phonemes, str) else phonemes
		ids = self._lut[cps]
		if (ids < 0).any():
			for c in np.unique(cps[ids < 0]):
				self._lut[c] = self._id("vocab", chr(int(c)))
			ids = self._lut[cps]
		self.shard.add(codes, self.speaker_id(group, speaker), self._id("languages", language.lower()), ids, text, name)

	def _close_shard(self):
		if self.shard is not None:
			self.m["shards"].append(self.shard.close())
			self.shard = None
			(self.root / "manifest.json").write_text(json.dumps(self.m))

	def close(self):
		self._close_shard()
		(self.root / "manifest.json").write_text(json.dumps(self.m))


class Store:
	"""Read side. Utterance i is global across shards. In memory: frames, speaker, language (int arrays), the
	speaker -> utterances grouping, and the manifest tables; everything else is memory-mapped."""
	def __init__(self, root):
		self.root = Path(root)
		self.m = json.loads((self.root / "manifest.json").read_text())
		self.shards = []
		for sh in self.m["shards"]:
			d = self.root / sh["name"]
			ld = lambda k: np.load(d / f"{k}.npy", mmap_mode="r")
			self.shards.append(dict(codes=np.memmap(d / "codes.u16", dtype=np.uint16, mode="r").reshape(-1, CODEC["groups"]),
				**{k: ld(k) for k in ("frame_offsets", "phonemes", "phoneme_offsets", "text", "text_offsets", "names", "name_offsets")}))
			last = self.shards[-1]
			if len(last["codes"]) != int(last["frame_offsets"][-1]) or len(last["frame_offsets"]) != sh["utterances"] + 1:
				raise RuntimeError(f"{d}: codes hold {len(last['codes'])} frames but the index ends at {int(last['frame_offsets'][-1])}")
		self.starts = np.cumsum([0] + [sh["utterances"] for sh in self.m["shards"]])
		self.frames = np.concatenate([np.diff(np.load(self.root / sh["name"] / "frame_offsets.npy")) for sh in self.m["shards"]]).astype(np.int32)
		self.speaker = np.concatenate([np.load(self.root / sh["name"] / "speaker.npy") for sh in self.m["shards"]])
		self.language = np.concatenate([np.load(self.root / sh["name"] / "language.npy") for sh in self.m["shards"]])
		self.group = np.asarray(self.m["speaker_group"], dtype=np.int32)[self.speaker]
		order = np.argsort(self.speaker, kind="stable")
		counts = np.bincount(self.speaker, minlength=len(self.m["speakers"]))
		self.speaker_utts, self.speaker_offsets = order.astype(np.int64), np.concatenate([[0], np.cumsum(counts)])
		# optional sidecars (see build_similar, build_exclude and noisy.py)
		side = lambda k: np.load(self.root / k, mmap_mode="r") if (self.root / k).exists() else None
		self.similar, self.exclude = side("similar.npy"), side("exclude.npy")
		self.noisy = None
		if (self.root / "noisy" / "source.npy").exists():
			d = self.root / "noisy"
			src = np.load(d / "source.npy")
			self.noisy = dict(codes=np.memmap(d / "codes.u16", dtype=np.uint16, mode="r").reshape(-1, CODEC["groups"]),
				offsets=np.load(d / "frame_offsets.npy"))
			# speaker -> its noised prompt (one per speaker), -1 if none
			self.noisy_of = np.full(len(self.m["speakers"]), -1, dtype=np.int64)
			self.noisy_of[self.speaker[src]] = np.arange(len(src))
			self.noisy_source = src

	def noisy_codes(self, k):
		"""Noised prompt k, [T, 8] uint16."""
		o = self.noisy["offsets"]
		return self.noisy["codes"][o[k] : o[k + 1]]

	def __len__(self):
		return int(self.starts[-1])

	def _loc(self, i):
		s = int(np.searchsorted(self.starts, i, side="right") - 1)
		return self.shards[s], i - int(self.starts[s])

	def codes(self, i):
		"""[T, 8] uint16 (a view into the memory map)."""
		sh, j = self._loc(i)
		return sh["codes"][sh["frame_offsets"][j] : sh["frame_offsets"][j + 1]]

	def phonemes(self, i):
		sh, j = self._loc(i)
		return np.asarray(sh["phonemes"][sh["phoneme_offsets"][j] : sh["phoneme_offsets"][j + 1]])

	def _bytes(self, i, key):
		sh, j = self._loc(i)
		o = sh[f"{key}_offsets"]
		return bytes(sh[key if key == "text" else "names"][o[j] : o[j + 1]]).decode()

	def text(self, i):
		return self._bytes(i, "text")

	def name(self, i):
		return self._bytes(i, "name")

	def speaker_key(self, i):
		"""'<group>/<speaker>', as the per-speaker metadata was keyed."""
		return f"{self.m['groups'][self.group[i]]}/{self.m['speakers'][self.speaker[i]]}"

	def utterances_of(self, speaker):
		return self.speaker_utts[self.speaker_offsets[speaker] : self.speaker_offsets[speaker + 1]]


# --- conversion from the per-speaker metadata JSONs and .nem artifacts ---

def _read_speaker(args):
	meta_file, data_root, group = args
	out, missing = [], 0
	try:
		# errors="replace": LibriVox/614.json has one corrupt byte inside an utterance's text
		meta = json.loads(Path(meta_file).read_bytes().decode("utf-8", errors="replace"))
	except (OSError, ValueError) as e:
		print(f"skipping unreadable metadata {meta_file}: {e}", flush=True)
		return group, Path(meta_file).stem, out, missing
	for uid, m in meta.items():
		ph = m.get("phonemes")
		if not ph:
			continue
		path = Path(data_root) / group / Path(meta_file).stem / f"{uid}.nem"
		try:
			codes = np.load(path, allow_pickle=True)[()]["codes"]
		except FileNotFoundError:
			missing += 1
			continue
		except Exception as e:  # corrupt artifacts (truncated pickles etc.) count as missing, but are reported
			print(f"skipping corrupt artifact {path}: {type(e).__name__}", flush=True)
			missing += 1
			continue
		out.append((uid, codes.astype(np.uint16), np.frombuffer(ph.encode("utf-32-le"), dtype=np.uint32), m.get("text", ""), m.get("language", "")))
	return group, Path(meta_file).stem, out, missing


def convert(meta_root, data_root, out, groups=None, workers=6, limit_speakers=None, max_inflight=32, resume=False):
	"""resume: reopen <out>, delete shard folders the manifest doesn't list (a shard that was never closed), and skip
	speakers the manifest already has. A speaker whose utterances straddled an unclosed shard keeps only its first part."""
	meta_root, out = Path(meta_root), Path(out)
	files = sorted(f for f in meta_root.rglob("*.json"))
	jobs = [(str(f), str(data_root), str(f.parent.relative_to(meta_root))) for f in files]
	if groups:
		jobs = [j for j in jobs if any(j[2] == g or j[2].startswith(g + "/") for g in groups)]
	if limit_speakers:
		jobs = jobs[:limit_speakers]
	if resume and (out / "manifest.json").exists():
		listed = {s["name"] for s in json.loads((out / "manifest.json").read_text())["shards"]}
		for d in out.glob("shard-*"):
			if d.name not in listed:
				print(f"removing unclosed shard {d.name}")
				shutil.rmtree(d)
	w = StoreWriter(out)
	if resume:
		before = len(jobs)
		jobs = [j for j in jobs if (w._ids["groups"].get(j[2]), Path(j[0]).stem) not in w._spk]
		print(f"resuming: {before - len(jobs)} speakers already converted, {len(jobs)} to go", flush=True)
	t0, n, missing, k = time.time(), 0, 0, 0
	# ProcessPoolExecutor raises if a reader dies, where Pool.imap hung silently; at most max_inflight speakers are
	# read ahead of the writer, and results are taken as they complete, so one slow speaker blocks nothing
	todo = iter(jobs)
	with ProcessPoolExecutor(workers) as ex:
		pending = set()
		def fill():
			while len(pending) < max_inflight and (j := next(todo, None)) is not None:
				pending.add(ex.submit(_read_speaker, j))
		fill()
		while pending:
			done, pending = wait(pending, return_when=FIRST_COMPLETED)
			for f in done:
				group, spk, items, miss = f.result()
				for uid, codes, ph, text, lang in items:
					w.add(codes, group, spk, lang, ph, text, uid)
				n += len(items); missing += miss; k += 1
				if k % 20000 == 0:
					print(f"{k}/{len(jobs)} speakers, {n} utterances, {missing} missing artifacts, {time.time() - t0:.0f}s", flush=True)
			fill()
	w.close()
	print(f"done: {n} utterances from {len(jobs)} speakers ({missing} missing artifacts) in {time.time() - t0:.0f}s", flush=True)


# --- sidecars: per-utterance arrays built after conversion, beside the shards (the shards are never rewritten) ---
#
#   <root>/similar.npy    int32 [utterances, SIMILAR_K]: global ids of each utterance's most similar same-speaker
#                         utterances (the metadata's `similar` lists, best first), -1 padded
#   <root>/exclude.npy    bool [utterances]: dropped from training (bad transcripts or labels); exclude.json says why
#   <root>/noisy/         codes.u16 + frame_offsets.npy + source.npy: noised copies of some utterances (prompts)

SIMILAR_K = 8


def _read_similar(args):
	meta_file, group = args
	try:
		meta = json.loads(Path(meta_file).read_bytes().decode("utf-8", errors="replace"))
	except (OSError, ValueError):
		return group, Path(meta_file).stem, None, None
	keys = list(meta.keys())
	return group, Path(meta_file).stem, keys, [meta[k].get("similar") or [] for k in keys]


def build_similar(root, meta_root, workers=6):
	"""The metadata's `similar` lists are indices into that speaker file's key order; this maps them to global
	utterance ids through each speaker's stored names (utterances the store dropped are skipped)."""
	st, meta_root = Store(root), Path(meta_root)
	out = np.full((len(st), SIMILAR_K), -1, dtype=np.int32)
	spk_id = {(st.m["groups"][g], s): i for i, (s, g) in enumerate(zip(st.m["speakers"], st.m["speaker_group"]))}
	jobs = [(str(f), str(f.parent.relative_to(meta_root))) for f in sorted(meta_root.rglob("*.json"))]
	jobs = [j for j in jobs if (j[1], Path(j[0]).stem) in spk_id]
	t0, n_lists = time.time(), 0
	with ProcessPoolExecutor(workers) as ex:
		for k, (group, spk, keys, sims) in enumerate(ex.map(_read_similar, jobs, chunksize=64)):
			if keys is None:
				continue
			utts = st.utterances_of(spk_id[(group, spk)])
			by_name = {st.name(int(u)): int(u) for u in utts}
			glob = [by_name.get(key, -1) for key in keys]
			for key, sim in zip(keys, sims):
				u = by_name.get(key)
				if u is None:
					continue
				ids = [glob[i] for i in sim if 0 <= i < len(glob) and glob[i] >= 0 and glob[i] != u][:SIMILAR_K]
				out[u, : len(ids)] = ids
				n_lists += bool(ids)
			if (k + 1) % 20000 == 0:
				print(f"{k + 1}/{len(jobs)} speakers, {time.time() - t0:.0f}s", flush=True)
	np.save(Path(root) / "similar.npy.tmp.npy", out)
	(Path(root) / "similar.npy.tmp.npy").replace(Path(root) / "similar.npy")
	print(f"similar: {n_lists}/{len(st)} utterances have a list, {time.time() - t0:.0f}s", flush=True)


# groups whose voice work is English: anything else in them is a transcription error (Whisper hearing a scream as
# Japanese) or an unreliable label (`auto`)
ENGLISH_GROUPS = ("GROUP", "GROUP", "GROUP", "GROUP", "GROUP", "GROUP", "GROUP", "GROUP", "GROUP", "GROUP", "GROUP", "GROUP",
	"GROUP", "GROUP", "GROUP", "GROUP", "GROUP")
# espeak's English voice reads CJK characters it can't pronounce as "Japanese letter" / "Chinese letter"
LETTER_READINGS = ("dʒˈæpəniːzlˌɛɾɚ", "tʃˈaɪniːzlˌɛɾɚ")


def build_exclude(root):
	import re
	st = Store(root)
	G, L, V = st.m["groups"], st.m["languages"], st.m["vocab"]
	lut = {c: i for i, c in enumerate(V)}
	reasons = {}
	# phoneme strings with letter readings: a byte search over each shard's phoneme array
	letter = np.zeros(len(st), dtype=bool)
	for s, sh in enumerate(st.shards):
		buf, off = np.asarray(sh["phonemes"]).tobytes(), np.asarray(sh["phoneme_offsets"])
		for reading in LETTER_READINGS:
			if any(c not in lut for c in reading):
				continue
			pat = re.escape(np.array([lut[c] for c in reading], dtype=np.uint16).tobytes())
			pos = [m.start() // 2 for m in re.finditer(pat, buf) if m.start() % 2 == 0]
			if pos:
				letter[int(st.starts[s]) + np.unique(np.searchsorted(off, pos, side="right") - 1)] = True
	reasons["letter readings in phonemes"] = letter
	# English-voiced groups: non-en labels, or text in a CJK / Hangul script
	eng = np.isin(st.group, [G.index(g) for g in ENGLISH_GROUPS if g in G])
	non_en = eng & (st.language != L.index("en"))
	cjk = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
	script = np.zeros(len(st), dtype=bool)
	for u in np.nonzero(eng & ~non_en)[0]:
		script[u] = bool(cjk.search(st.text(int(u))))
	reasons["non-English label in an English-voiced group"] = non_en
	reasons["CJK text in an English-voiced group"] = script
	# GROUP is Japanese: items labelled anything else were phonemized in the wrong language
	if "GROUP" in G:
		reasons["GROUP item not labelled ja"] = (st.group == G.index("GROUP")) & (st.language != L.index("ja"))
	exclude = np.zeros(len(st), dtype=bool)
	for v in reasons.values():
		exclude |= v
	np.save(Path(root) / "exclude.npy", exclude)
	summary = {k: int(v.sum()) for k, v in reasons.items()} | dict(total=int(exclude.sum()),
		hours=round(float(st.frames[exclude].sum()) / (CODEC["sample_rate"] / CODEC["samples_per_frame"]) / 3600, 2))
	(Path(root) / "exclude.json").write_text(json.dumps(summary, indent=1))
	print(summary, flush=True)


def _load_codes(path):
	codes = np.load(path, allow_pickle=True)[()]["codes"]
	while codes.ndim > 2 and codes.shape[0] == 1:
		codes = codes[0]
	return np.ascontiguousarray(codes.T, dtype=np.uint16)


def repair_codes(root, data_root, shard_names=None, workers=3, chunk=20000):
	"""Rewrites the codes and frame offsets of shards whose codes file and index disagree (shard_names, or every such
	shard), re-reading each utterance's artifact by its recorded group, speaker and name. Phonemes, text and names are
	kept. Written for the first conversion's [1, 8, T] artifacts, which were indexed as one frame."""
	root = Path(root)
	m = json.loads((root / "manifest.json").read_text())
	for k, sh in enumerate(m["shards"]):
		d = root / sh["name"]
		if shard_names is not None and sh["name"] not in shard_names:
			continue
		if shard_names is None and (d / "codes.u16").stat().st_size == sh["frames"] * CODEC["groups"] * 2:
			continue
		t0 = time.time()
		spk = np.load(d / "speaker.npy")
		noff, names = np.load(d / "name_offsets.npy"), bytes(np.load(d / "names.npy"))
		paths = [str(Path(data_root) / m["groups"][m["speaker_group"][s]] / m["speakers"][s] / f"{names[noff[j]:noff[j + 1]].decode()}.nem")
			for j, s in enumerate(spk.tolist())]
		frames = [0]
		with open(d / "codes.u16.tmp", "wb") as f, ProcessPoolExecutor(workers) as ex:
			for i in range(0, len(paths), chunk):
				for codes in ex.map(_load_codes, paths[i : i + chunk], chunksize=256):
					f.write(codes.tobytes()); frames.append(frames[-1] + len(codes))
		if (d / "codes.u16.tmp").stat().st_size != frames[-1] * CODEC["groups"] * 2:
			raise RuntimeError(f"{d}: rebuilt codes file and offsets disagree")
		(d / "codes.u16.tmp").replace(d / "codes.u16")
		np.save(d / "frame_offsets.npy", np.asarray(frames, dtype=np.int64))
		m["shards"][k]["frames"] = frames[-1]
		(root / "manifest.json").write_text(json.dumps(m))
		print(f"repaired {sh['name']}: {len(paths)} utterances, {frames[-1]} frames, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	sub = ap.add_subparsers(dest="cmd", required=True)
	c = sub.add_parser("convert")
	c.add_argument("--meta", type=Path, required=True)
	c.add_argument("--data", type=Path, required=True)
	c.add_argument("--out", type=Path, required=True)
	c.add_argument("--groups", nargs="*", default=None, help="e.g. LibriTTS-R Emilia/EN (default: all)")
	c.add_argument("--workers", type=int, default=6)
	c.add_argument("--limit-speakers", type=int, default=None)
	c.add_argument("--resume", action="store_true", help="continue an interrupted conversion into the same --out")
	r = sub.add_parser("repair-codes", help="rebuild the codes of shards whose codes file and index disagree")
	r.add_argument("--store", type=Path, required=True)
	r.add_argument("--data", type=Path, required=True)
	r.add_argument("--workers", type=int, default=3)
	s = sub.add_parser("similar", help="build similar.npy from the metadata's similar lists")
	s.add_argument("--store", type=Path, required=True)
	s.add_argument("--meta", type=Path, required=True)
	s.add_argument("--workers", type=int, default=6)
	e = sub.add_parser("exclude", help="build exclude.npy (bad transcripts and labels)")
	e.add_argument("--store", type=Path, required=True)
	args = ap.parse_args()
	if args.cmd == "convert":
		convert(args.meta, args.data, args.out, args.groups, args.workers, args.limit_speakers, resume=args.resume)
	elif args.cmd == "repair-codes":
		repair_codes(args.store, args.data, workers=args.workers)
	elif args.cmd == "similar":
		build_similar(args.store, args.meta, args.workers)
	else:
		build_exclude(args.store)
