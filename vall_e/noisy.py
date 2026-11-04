# Noised prompts for the v3 store, baked offline: NeMo codes can't be mixed, so each chosen utterance is decoded, mixed
# with a noise clip at a random SNR and re-encoded.
#
#   <root>/noise/   the noise clips (Other/Noise, which has no transcripts), as a text-less shard
#   <root>/noisy/   codes.u16 [frames, 8] + frame_offsets.npy, source.npy (global utterance id), snr.npy (dB)
#
#   python -m vall_e.noisy import-noise --store <root> --meta training/llama-nemo/metadata --data training/llama-nemo/data
#   python -m vall_e.noisy bake --store <root>
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .store import CODEC, ShardWriter, Store

MAX_S = 10.0  # the longest training prompt


def import_noise(root, meta_root, data_root, group="Other/Noise"):
	meta = json.loads((Path(meta_root) / f"{group}.json").read_text())
	w = ShardWriter(Path(root) / "noise")
	n = 0
	for uid in meta:
		path = Path(data_root) / group / f"{uid}.nem"
		try:
			codes = np.load(path, allow_pickle=True)[()]["codes"]
		except Exception as e:
			print(f"skipping {path}: {type(e).__name__}")
			continue
		w.add(codes, 0, 0, np.zeros(0, dtype=np.uint16), "", uid)
		n += 1
	info = w.close()
	print(f"noise: {n} clips, {info['frames'] / (CODEC['sample_rate'] / CODEC['samples_per_frame']):.0f} s")


def load_noise(root):
	d = Path(root) / "noise"
	codes = np.memmap(d / "codes.u16", dtype=np.uint16, mode="r").reshape(-1, CODEC["groups"])
	off = np.load(d / "frame_offsets.npy")
	return [codes[off[i] : off[i + 1]] for i in range(len(off) - 1)]


def choose(st, seed=0, min_s=3.0):
	"""One utterance per speaker (the source of its noised prompt): a random non-excluded one of at least min_s."""
	fps = CODEC["sample_rate"] / CODEC["samples_per_frame"]
	exclude = np.load(st.root / "exclude.npy") if (st.root / "exclude.npy").exists() else np.zeros(len(st), dtype=bool)
	ok = (st.frames >= int(min_s * fps)) & ~exclude
	rng = np.random.default_rng(seed)
	out = []
	for s in range(len(st.m["speakers"])):
		utts = st.utterances_of(s)
		utts = utts[ok[utts]]
		if len(utts):
			out.append(int(utts[rng.integers(len(utts))]))
	return np.asarray(out, dtype=np.int64)


@torch.no_grad()
def bake(root, batch=32, snr=(3.0, 20.0), seed=0, limit=None, speakers=None, amp=False, device="cuda"):
	"""speakers: bake prompts for this many random speakers (default all); limit: only the first `limit` of those, into
	noisy-test (a quick check). amp: run the codec in bf16."""
	from .emb.nemo import AudioCodecModel
	root = Path(root)
	st = Store(root)
	fps = CODEC["sample_rate"] / CODEC["samples_per_frame"]
	max_f, spf = int(MAX_S * fps), CODEC["samples_per_frame"]
	codec = AudioCodecModel.from_pretrained(CODEC["name"]).to(device).eval()
	dec = lambda c: codec.decode(tokens=c, tokens_len=torch.tensor([c.shape[-1]] * len(c), device=device))[0].float()

	# clips run from under a second to minutes; MAX_S of each (looped as needed) is plenty to draw from
	with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
		noise = [dec(torch.from_numpy(c[:max_f].astype(np.int64)).t()[None].to(device))[0].float() for c in load_noise(root)]
	noise = [n / n.pow(2).mean().sqrt().clamp(min=1e-5) for n in noise if len(n) > spf]
	src = choose(st, seed)
	rng = np.random.default_rng(seed)
	if speakers:
		src = np.sort(rng.choice(src, size=min(speakers, len(src)), replace=False))
	if limit:
		src = rng.permutation(src)[:limit]
	# crop windows (start, length) of up to MAX_S, then batch by length so little is padded
	starts, lens = [], []
	for u in src:
		T = int(st.frames[u]); n = min(T, max_f)
		starts.append(int(rng.integers(T - n + 1))); lens.append(n)
	starts, lens = np.asarray(starts), np.asarray(lens)
	order = np.argsort(lens, kind="stable")

	out = root / "noisy.tmp"
	out.mkdir(exist_ok=True)
	f = open(out / "codes.u16", "wb")
	frames, source, snrs = [0], [], []
	t0 = time.time()
	for k in range(0, len(order), batch):
		idx = order[k : k + batch]
		T = -(-int(lens[idx].max()) // 64) * 64  # few distinct shapes, so MIOpen tunes each once
		codes = torch.zeros(len(idx), CODEC["groups"], T, dtype=torch.long)
		for r, i in enumerate(idx):
			c = st.codes(int(src[i]))[starts[i] : starts[i] + lens[i]]
			codes[r, :, : lens[i]] = torch.from_numpy(c.astype(np.int64)).t()
		tl = torch.as_tensor(lens[idx], device=device)
		with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
			wav, wl = codec.decode(tokens=codes.to(device), tokens_len=tl)
		wav = wav.float()
		# noise at a random SNR against each utterance's own power (over its valid samples); clips loop to length
		s = rng.uniform(*snr, size=len(idx))
		for r in range(len(idx)):
			L = int(wl[r])
			clean = wav[r, :L]
			nz = noise[rng.integers(len(noise))]
			nz = nz.repeat(L // len(nz) + 2)
			o = int(rng.integers(len(nz) - L))
			p = clean.pow(2).mean().clamp(min=1e-8)
			mix = clean + nz[o : o + L] * (p / 10 ** (s[r] / 10)).sqrt()
			wav[r, :L] = mix / mix.abs().max().clamp(min=1.0)  # no clipping
			wav[r, L:] = 0
		with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
			toks, toks_len = codec.encode(audio=wav, audio_len=wl)
		toks = toks.cpu().numpy().astype(np.uint16)
		for r, i in enumerate(idx):
			n = min(int(toks_len[r]), int(lens[i]))
			f.write(np.ascontiguousarray(toks[r, :, :n].T).tobytes())
			frames.append(frames[-1] + n); source.append(int(src[i])); snrs.append(float(s[r]))
		if (k // batch) % 200 == 0:
			done = k + len(idx)
			print(f"{done}/{len(order)} prompts, {(frames[-1] / fps) / 3600:.1f} h, {time.time() - t0:.0f}s "
				f"({(frames[-1] / fps) / max(time.time() - t0, 1e-6):.0f}x realtime)", flush=True)
	f.close()
	if (out / "codes.u16").stat().st_size != frames[-1] * CODEC["groups"] * 2:
		raise RuntimeError("noisy codes file and offsets disagree")
	np.save(out / "frame_offsets.npy", np.asarray(frames, dtype=np.int64))
	np.save(out / "source.npy", np.asarray(source, dtype=np.int64))
	np.save(out / "snr.npy", np.asarray(snrs, dtype=np.float32))
	final = root / ("noisy" if not limit else "noisy-test")
	if final.exists():
		raise RuntimeError(f"{final} exists; move it aside first")
	out.rename(final)
	print(f"done: {len(source)} noised prompts, {(frames[-1] / fps) / 3600:.1f} h in {time.time() - t0:.0f}s -> {final}")


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	sub = ap.add_subparsers(dest="cmd", required=True)
	a = sub.add_parser("import-noise")
	a.add_argument("--store", type=Path, required=True)
	a.add_argument("--meta", type=Path, required=True)
	a.add_argument("--data", type=Path, required=True)
	b = sub.add_parser("bake")
	b.add_argument("--store", type=Path, required=True)
	b.add_argument("--batch", type=int, default=32)
	b.add_argument("--limit", type=int, default=None, help="only this many prompts, into <store>/noisy-test (speed and listening check)")
	b.add_argument("--speakers", type=int, default=None, help="one prompt for each of this many random speakers (default: every speaker)")
	b.add_argument("--amp", action="store_true", help="bf16 codec")
	args = ap.parse_args()
	if args.cmd == "import-noise":
		import_noise(args.store, args.meta, args.data)
	else:
		bake(args.store, args.batch, limit=args.limit, speakers=args.speakers, amp=args.amp)
