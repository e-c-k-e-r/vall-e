# Ingesting Hugging Face voice datasets into a store (store.py) of their own: the audio encoded with the NeMo codec,
# the text phonemized against the main store's vocabulary (so phoneme ids line up across stores; phonemes the main
# store doesn't have are dropped and counted), streamed one parquet file at a time (the next downloads while the
# current one is encoded) and resumable per file.
#
#   python -m vall_e.ingest genshin --out <new store root> --vocab-from <main store root> [--languages en ja] [--files 0 75]
#
# genshin: simon3000/genshin-voice (Genshin Impact voice lines: speaker, transcription and language from the game's
# own tables; 75 parquet files, the languages mixed within each). Kept: lines with a speaker and a transcription in the
# line's language, 0.6-30 s; markup cleaned; lines with placeholders the audio can't contain ({NICKNAME}, player-
# gendered {M#..}{F#..} variants, ...) dropped.
import argparse
import io
import json
import os
import re
import threading
from pathlib import Path

os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from .store import CODEC, StoreWriter

FPS = CODEC["sample_rate"] / CODEC["samples_per_frame"]
GENSHIN = dict(repo="simon3000/genshin-voice", n_files=75, languages={"English(US)": "en", "Japanese": "ja", "Chinese": "zh", "Korean": "ko"})
SCRIPT = {"en": r"[A-Za-z]", "ja": r"[぀-ヿ]", "zh": r"[一-鿿]", "ko": r"[가-힯]"}
DROP = re.compile(r"\{(NICKNAME|REALNAME|PLAYERAVATAR|SPRITE_PRESET|M#|F#)")


def clean_genshin(text):
	"""The spoken text of a transcription, or None if it holds something the audio can't (or might not) contain."""
	if not text or DROP.search(text):
		return None
	t = text.lstrip("#")
	t = re.sub(r"\{RUBY#\[[^\]]*\][^}]*\}", "", t)  # ruby annotations: the base text stays
	t = re.sub(r"<[^>]+>", "", t).replace("\\n", " ").replace("\n", " ")
	if "{" in t or "}" in t:
		return None
	t = re.sub(r"\s+", " ", t).strip()
	return t or None


def _download(repo, i, n, dest):
	from huggingface_hub import hf_hub_download
	name = f"data/train-{i:05d}-of-{n:05d}.parquet"
	return Path(hf_hub_download(repo, name, repo_type="dataset", local_dir=dest))


def ingest_genshin(out, vocab_from, languages=("en", "ja"), files=(0, 75), batch=12, device="cuda", max_row_groups=None, keep_download=False, budget_s=60.0):
	import pyarrow.parquet as pq
	import soundfile as sf
	import torchaudio
	from .emb.g2p import encode as g2p
	from .evalset import load_codec
	os.environ["HF_HUB_OFFLINE"] = "0"
	out = Path(out); out.mkdir(parents=True, exist_ok=True)
	main = json.loads((Path(vocab_from) / "manifest.json").read_text())
	if not (out / "manifest.json").exists():
		# the main store's vocabulary and languages, so ids line up; groups and speakers start empty
		(out / "manifest.json").write_text(json.dumps(dict(codec=CODEC, shards=[], groups=[], speakers=[], speaker_group=[],
			languages=main["languages"], vocab=main["vocab"])))
	known = set(main["vocab"])
	state_path = out / "ingest.json"
	state = json.loads(state_path.read_text()) if state_path.exists() else dict(done=[], stats={})
	codec = load_codec(device)
	writer = StoreWriter(out)
	lang_of = {k: v for k, v in GENSHIN["languages"].items() if v in languages}
	todo = [i for i in range(*files) if i not in state["done"]]
	dl_dir = out / "download"
	prefetch = {}

	def fetch(i):
		prefetch[i] = _download(GENSHIN["repo"], i, GENSHIN["n_files"], dl_dir)

	th = threading.Thread(target=fetch, args=(todo[0],)) if todo else None
	if th: th.start()
	for k, i in enumerate(todo):
		th.join()
		path = prefetch.pop(i)
		if k + 1 < len(todo):
			th = threading.Thread(target=fetch, args=(todo[k + 1],)); th.start()
		f = pq.ParquetFile(path)
		stats = dict(rows=0, kept=0, no_speaker=0, no_text=0, wrong_script=0, markup=0, duration=0, phonemes_dropped=0)
		for rg in range(min(f.num_row_groups, max_row_groups or f.num_row_groups)):
			t = f.read_row_group(rg).to_pydict()
			rows = []
			for j in range(len(t["language"])):
				stats["rows"] += 1
				lang = lang_of.get(t["language"][j])
				if lang is None:
					continue
				if not t["speaker"][j]:
					stats["no_speaker"] += 1; continue
				raw = t["transcription"][j]
				if not raw:
					stats["no_text"] += 1; continue
				text = clean_genshin(raw)
				if text is None:
					stats["markup"] += 1; continue
				if not re.search(SCRIPT[lang], text):
					stats["wrong_script"] += 1; continue
				rows.append((j, lang, text))
			# decode, resample, then encode in batches of similar length
			items = []
			for j, lang, text in rows:
				a = t["audio"][j]
				w, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=True)
				w = torch.from_numpy(w.mean(1))
				if sr != CODEC["sample_rate"]:
					w = torchaudio.functional.resample(w, sr, CODEC["sample_rate"])
				s = len(w) / CODEC["sample_rate"]
				if not 0.6 <= s <= 30.0:
					stats["duration"] += 1; continue
				items.append((w, j, lang, text))
			items.sort(key=lambda x: len(x[0]))
			# batches of at most `batch` items and budget_s seconds of padded audio (12 lines of 30 s ran out of 24 GB)
			parts, cur = [], []
			for x in items:
				if cur and (len(cur) >= batch or len(x[0]) * (len(cur) + 1) > budget_s * CODEC["sample_rate"]):
					parts.append(cur); cur = []
				cur.append(x)
			if cur:
				parts.append(cur)
			while parts:
				part = parts.pop(0)
				T = max(len(x[0]) for x in part)
				T = -(-T // (CODEC["samples_per_frame"] * 87)) * CODEC["samples_per_frame"] * 87  # ~1 s buckets: few conv shapes
				audio = None
				try:
					audio = torch.stack([torch.nn.functional.pad(x[0], (0, T - len(x[0]))) for x in part]).to(device)
					lens = torch.tensor([len(x[0]) for x in part], device=device)
					with torch.no_grad():
						codes, n = codec.encode(audio=audio, audio_len=lens)
				except torch.OutOfMemoryError:
					del audio
					torch.cuda.empty_cache()
					if len(part) == 1:
						raise
					parts[:0] = [part[: len(part) // 2], part[len(part) // 2 :]]  # halve and retry
					continue
				codes, n = codes.cpu().numpy(), n.cpu().numpy()
				for (w, j, lang, text), c, nn in zip(part, codes, n):
					ph = g2p(text, language=lang)
					kept = "".join(ch for ch in ph if ch in known)
					stats["phonemes_dropped"] += len(ph) - len(kept)
					if not kept:
						continue
					writer.add(c[:, : int(nn)], f"Genshin/{lang.upper()}", t["speaker"][j], lang, kept, text, t["inGameFilename"][j] or f"{i}:{rg}:{j}")
					stats["kept"] += 1
		writer._close_shard()
		state["done"].append(i); state["stats"][str(i)] = stats
		state_path.write_text(json.dumps(state, indent=1))
		if not keep_download:
			path.unlink()  # the parquet: the codes are what we keep
		print(f"file {i}: {stats}", flush=True)
	writer.close()


def main():
	ap = argparse.ArgumentParser()
	sub = ap.add_subparsers(dest="cmd", required=True)
	g = sub.add_parser("genshin")
	g.add_argument("--out", type=Path, required=True)
	g.add_argument("--vocab-from", type=Path, required=True, help="the main store: its phoneme vocabulary and languages are reused")
	g.add_argument("--languages", nargs="*", default=["en", "ja"])
	g.add_argument("--files", type=int, nargs=2, default=[0, 75], help="parquet file range [start, end)")
	g.add_argument("--batch", type=int, default=12)
	g.add_argument("--max-row-groups", type=int, default=None, help="smoke tests: only the first row groups of each file")
	g.add_argument("--keep-download", action="store_true")
	args = ap.parse_args()
	ingest_genshin(args.out, args.vocab_from, tuple(args.languages), tuple(args.files), args.batch, max_row_groups=args.max_row_groups, keep_download=args.keep_download)


if __name__ == "__main__":
	main()
