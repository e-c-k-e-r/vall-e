# Repair test: mask a small fraction of a real utterance's positions, let the model fill them in one pass with both
# neighbours known, decode, and measure log-mel error per band on the repaired frames against the real recording.
# With nearly all context given, capacity matters little, so damage that remains (especially to upper harmonics)
# points at the per-scalar head. A copy-the-previous-position fill is the naive baseline.
import argparse
import json
import random
from pathlib import Path

import librosa
import numpy as np
import torch

from . import fsq
from .data import Utterances, to_patches
from .model import V3, flow_sample, metrics, snap

SR, HOP = 44100, 256
BANDS = [(0, 1000), (1000, 4000), (4000, 8000), (8000, 22050)]


def logmel(y):
	S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=2048, hop_length=HOP, n_mels=160, fmax=22050)
	return np.log10(S + 1e-6), librosa.mel_frequencies(n_mels=160, fmax=22050)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--ckpt", type=Path, required=True)
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--n", type=int, default=8)
	ap.add_argument("--ratio", type=float, default=0.15)
	ap.add_argument("--cfg", type=float, default=2.5)
	ap.add_argument("--device", default="cpu")
	ap.add_argument("--temperature", type=float, default=1.0, help="flow head: scale of the starting noise")
	args = ap.parse_args()
	dev = args.device
	args.out.mkdir(parents=True, exist_ok=True)

	st = torch.load(args.ckpt, map_location=dev, weights_only=False)
	model = V3(**st["config"]).to(dev).eval()
	model.load_state_dict(st["model"])
	from .data import StoreUtterances
	index = json.loads((args.ckpt.parent / "index.json").read_text()) if not st.get("store") else None
	val = StoreUtterances(st["store"], "val", seed=1, groups=st.get("groups"), languages=st.get("languages")) if st.get("store") else Utterances(index, "val", seed=1)
	items = [val[i] for i in random.Random(0).sample(range(len(val)), args.n)]

	from ..emb.codecs.nemo import AudioCodecModel
	codec = AudioCodecModel.from_pretrained("nvidia/audio-codec-44khz").to(dev).eval()

	def decode(patches, n_frames):
		codes = fsq.digits_to_codes(patches.reshape(-1, fsq.SCALARS)[:n_frames]).unsqueeze(0)
		with torch.no_grad():
			wav, _ = codec.decode(tokens=codes, tokens_len=torch.tensor([codes.shape[-1]]))
		return wav[0].float().numpy()

	rng = random.Random(0)
	err = {k: {b: [] for b in BANDS} for k in ("model", "copy_prev")}
	head = {"scalar_acc": [], "group_acc": [], "mean_abs_err": []}
	for i, it in enumerate(items):
		a = to_patches(it["target"], model.patch)
		n = len(a)
		# interior positions only, so each masked position has real neighbours on both sides
		idx = sorted(rng.sample(range(1, n - 1), max(1, round(args.ratio * n))))
		m = torch.zeros(n, dtype=torch.bool); m[idx] = True
		with torch.no_grad():
			text, prompt = it["text"], to_patches(it["prompt"], model.patch)
			h = model.encode([text, text[:0]], [prompt, prompt[:0]], [a, a], [m, m])
			if model.head_type == "flow":
				vals = flow_sample(model.flow, h[:n][m], h[n:][m], cfg=args.cfg, temperature=args.temperature)
				pred = torch.zeros_like(a); pred[m] = snap(vals, model.patch)
			else:
				out = model.logits(h)
				logits = out[n:] + (out[:n] - out[n:]) * args.cfg if args.cfg > 0 else out[:n]
				# -inf - -inf on the padded levels is nan; restore -inf so argmax never picks a level a scalar doesn't have
				pred = logits.masked_fill(~torch.isfinite(logits), float("-inf")).argmax(-1)
		for k, v in metrics(pred[m], a[m]).items():
			head[k].append(v)
		fills = dict(model=torch.where(m[:, None], pred, a), copy_prev=a.clone())
		for j in idx:
			fills["copy_prev"][j] = fills["copy_prev"][j - 1]
		ref = decode(a, len(it["target"]))
		R, freqs = logmel(ref)
		# mel frames covered by masked positions (patch * 512 samples each)
		per = model.patch * 512 // HOP
		frames = np.concatenate([np.arange(j * per, (j + 1) * per) for j in idx])
		frames = frames[frames < R.shape[1]]
		for k, f in fills.items():
			y = decode(f, len(it["target"]))
			if k == "model" and i < 4:
				import soundfile as sf
				sf.write(args.out / f"{i:02d}_repaired.wav", y, SR); sf.write(args.out / f"{i:02d}_ref.wav", ref, SR)
			G, _ = logmel(y)
			L = min(R.shape[1], G.shape[1])
			fr = frames[frames < L]
			for lo, hi in BANDS:
				band = (freqs >= lo) & (freqs < hi)
				err[k][(lo, hi)].append(float(np.abs(R[band][:, fr] - G[band][:, fr]).mean()))
	res = dict(head={k: float(np.mean(v)) for k, v in head.items()},
		logmel_l1={k: {f"{lo}-{hi}": float(np.mean(v[(lo, hi)])) for lo, hi in BANDS} for k, v in err.items()})
	print(json.dumps(res, indent=1))
	(args.out / "repair.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
	main()
