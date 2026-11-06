# REPA targets (train.py --repa-feats): frozen speech-model features of a fixed subset of training utterances, baked
# once. Decoding the audio and running HuBERT on the fly made a training step ~20x slower on ROCm, and MIOpen's fast
# mode, which fixes that, segfaults in training under torch 2.7, so the targets are computed here, inference only.
#
#   python -m vall_e.repa bake --store <root> --out training/repa-mhubert --count 200000 [groups / filters as train.py]
#
# Writes ids.npy (store utterance indices, sorted), offsets.npy (patch offsets per utterance), feats.npy ([patches, dim]
# memmap: mHuBERT-147 layer-9 features at the model's patch rate, PCA-reduced) and pca.npz.
import argparse
import os
from pathlib import Path

os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F

from . import fsq
from .data import FPS, StoreUtterances


@torch.no_grad()
def bake(store, out, count, groups=None, min_rate=13, min_rate_groups=("LibriVox", "Audiobooks"), patch=2, dim=128,
		model_name="utter-project/mHuBERT-147", layer=9, max_s=12.0, seed=0, chunk=4, device="cuda"):
	import torchaudio
	from transformers import HubertModel
	from .evalset import load_codec
	out = Path(out); out.mkdir(parents=True, exist_ok=True)
	ds = StoreUtterances(store, "train", groups=groups, min_s=2.0, max_s=max_s, min_rate=min_rate, min_rate_groups=list(min_rate_groups))
	st = ds.store
	frames = st.frames[ds.items]
	ok = ds.items[(frames >= ds.min_f) & (frames <= ds.max_f)]
	ids = np.sort(np.random.default_rng(seed).choice(ok, size=min(count, len(ok)), replace=False))
	patches = -(-st.frames[ids] // patch)
	offsets = np.concatenate([[0], np.cumsum(patches)]).astype(np.int64)
	np.save(out / "ids.npy", ids); np.save(out / "offsets.npy", offsets)
	codec = load_codec(device)
	ssl = HubertModel.from_pretrained(model_name).to(device).eval().half()

	def feats_of(us):
		cs = [torch.from_numpy(np.asarray(st.codes(int(u))).astype(np.int64)).t() for u in us]  # [8, T]
		T = -(-max(c.shape[1] for c in cs) // 87) * 87  # 1 s buckets: few distinct conv shapes
		codes = torch.stack([F.pad(c, (0, T - c.shape[1])) for c in cs]).to(device)
		wav, wl = codec.decode(tokens=codes, tokens_len=torch.tensor([c.shape[1] for c in cs], device=device))
		wav16 = torchaudio.functional.resample(wav.float(), 44100, 16000)
		hs = ssl(wav16.half(), output_hidden_states=True).hidden_states[layer].float()
		res = []
		for j, c in enumerate(cs):
			n = max(1, int(wl[j]) * 16000 // 44100 // 320)
			La = -(-c.shape[1] // patch)
			res.append(F.interpolate(hs[j, :n].t()[None], size=La, mode="linear", align_corners=False)[0].t())
		return res

	# PCA from the first 2000 utterances
	sample = torch.cat([f for i in range(0, min(2000, len(ids)), chunk) for f in feats_of(ids[i : i + chunk])])
	mean = sample.mean(0)
	_, _, V = torch.linalg.svd(sample - mean, full_matrices=False)
	W = V[:dim].t().contiguous()  # [768, dim]
	scale = ((sample - mean) @ W).std(0).mean()
	np.savez(out / "pca.npz", mean=mean.cpu().numpy(), W=W.cpu().numpy(), scale=float(scale))
	print(f"PCA: {dim} of {sample.shape[1]} dims keep {float(((sample - mean) @ W).var(0).sum() / (sample - mean).var(0).sum()):.1%} of the variance", flush=True)
	mm = np.lib.format.open_memmap(out / "feats.npy", mode="w+", dtype=np.float16, shape=(int(offsets[-1]), dim))
	for i in range(0, len(ids), chunk):
		for j, f in enumerate(feats_of(ids[i : i + chunk])):
			k = i + j
			mm[offsets[k] : offsets[k + 1]] = (((f - mean) @ W) / scale).cpu().numpy().astype(np.float16)
		if i % 10000 == 0:
			print(f"{i}/{len(ids)}", flush=True)
	mm.flush()
	(out / "done").write_text("ok\n")
	print(f"wrote {len(ids)} utterances, {int(offsets[-1])} patches to {out}")


def main():
	ap = argparse.ArgumentParser()
	sub = ap.add_subparsers(dest="cmd", required=True)
	b = sub.add_parser("bake")
	b.add_argument("--store", required=True)
	b.add_argument("--out", type=Path, required=True)
	b.add_argument("--count", type=int, default=200_000)
	b.add_argument("--groups", nargs="*", default=None)
	b.add_argument("--dim", type=int, default=128)
	args = ap.parse_args()
	bake(args.store, args.out, args.count, groups=args.groups, dim=args.dim)


if __name__ == "__main__":
	main()
