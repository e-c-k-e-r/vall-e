# Prosody measurements for the eval set: pitch (CREPE), energy, pauses and rate per utterance, and how a generated
# contour agrees with a reference reading of the same sentence after aligning the two (DTW over MFCCs).
#
# Two kinds of comparison (evalset.py prosody):
# - against the prompt, the speaker's way of talking: register (mean log F0) and range (its spread) in semitones,
#   articulation rate (units per second of speech, pauses excluded);
# - against the ground truth, the speaker's own reading of this sentence: duration, pause share, and the correlation of
#   the aligned pitch and energy contours. A sentence has many good readings, so these are noisy per item and meant for
#   paired comparisons between models.
from functools import cache

import numpy as np
import torch
from ..audio import load as load_audio
import torchaudio

SR = 16000
HOP = 160  # 10 ms frames


def load(path):
	wav, sr = load_audio(path)
	return torchaudio.functional.resample(wav.mean(0, keepdim=True), sr, SR)


@torch.no_grad()
def features(path, device="cuda"):
	"""Per-frame (10 ms) log F0 in semitones (nan where unvoiced), energy in dB, speech activity, and MFCCs [13, T]."""
	import librosa
	import torchcrepe
	wav = load(path)
	f0, per = torchcrepe.predict(wav, SR, HOP, fmin=50.0, fmax=800.0, model="full", return_periodicity=True,
		batch_size=1024, device=device, pad=True)
	per = torchcrepe.filter.median(per, 3)
	f0 = f0[0].cpu().numpy(); per = per[0].cpu().numpy()
	y = wav[0].numpy()
	n = len(f0)
	frames = np.lib.stride_tricks.sliding_window_view(np.pad(y, (200, 200 + HOP)), 400)[::HOP][:n]
	energy = 10 * np.log10(np.maximum((frames ** 2).mean(-1), 1e-10))
	active = energy > np.percentile(energy, 95) - 35.0
	voiced = (per > 0.5) & active
	semi = np.where(voiced, 12 * np.log2(np.maximum(f0, 1.0) / 100.0), np.nan)
	mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, n_fft=400, hop_length=HOP)[:, :n]
	mfcc = mfcc - mfcc.mean(1, keepdims=True)
	return dict(semi=semi.astype(np.float32), energy=energy.astype(np.float32), active=active, mfcc=mfcc.astype(np.float32))


def summary(feat, min_pause=0.1):
	"""Utterance-level: register and range (semitones), the speech span and its pause share (silences of at least
	min_pause s between the first and last active frame), seconds of speech without the pauses."""
	act = feat["active"]
	idx = np.nonzero(act)[0]
	if len(idx) < 2:
		return dict(register=np.nan, range=np.nan, span=0.0, pause=0.0, speech=0.0)
	inner = act[idx[0] : idx[-1] + 1]
	# silent runs inside the span
	pause, run = 0, 0
	for a in inner:
		if not a:
			run += 1
		else:
			if run * HOP / SR >= min_pause:
				pause += run
			run = 0
	span = len(inner) * HOP / SR
	s = feat["semi"][~np.isnan(feat["semi"])]
	return dict(register=float(np.median(s)) if len(s) else np.nan, range=float(np.std(s)) if len(s) > 1 else np.nan,
		span=span, pause=pause * HOP / SR / span, speech=span - pause * HOP / SR)


def contour(a, b):
	"""Agreement of two readings of the same text after DTW alignment: Pearson correlation of the pitch contours (each
	relative to its own mean, so register differences don't count) over frames voiced in both, their RMS difference in
	cents, and the correlation of the energy contours over frames active in both."""
	import librosa
	_, wp = librosa.sequence.dtw(X=a["mfcc"], Y=b["mfcc"], metric="euclidean")
	i, j = wp[::-1, 0], wp[::-1, 1]
	sa, sb = a["semi"][i], b["semi"][j]
	both = ~np.isnan(sa) & ~np.isnan(sb)
	res = dict(f0_corr=np.nan, f0_rmse=np.nan, energy_corr=np.nan)
	if both.sum() > 10:
		x, y = sa[both] - np.nanmean(a["semi"]), sb[both] - np.nanmean(b["semi"])
		res["f0_corr"] = float(np.corrcoef(x, y)[0, 1])
		res["f0_rmse"] = float(np.sqrt(np.mean((x - y) ** 2)) * 100)
	act = a["active"][i] & b["active"][j]
	if act.sum() > 10:
		res["energy_corr"] = float(np.corrcoef(a["energy"][i][act], b["energy"][j][act])[0, 1])
	return res


def compare_to_refs(feat, units, prompt, gt):
	"""The metrics of one generated utterance (its features, number of text units) against the prompt's summary
	(with its own rate) and the ground truth's features and summary. Lower is better except the correlations."""
	s = summary(feat)
	rate = units / max(s["speech"], 0.1)
	out = dict(register_diff=abs(s["register"] - prompt["register"]), range_diff=abs(s["range"] - prompt["range"]),
		rate_vs_prompt=abs(float(np.log2(rate / prompt["rate"]))) if prompt.get("rate") else np.nan)
	if gt is not None:
		g = gt["summary"]
		out |= dict(dur_vs_gt=abs(float(np.log2(max(s["span"], 0.1) / max(g["span"], 0.1)))), pause_vs_gt=abs(s["pause"] - g["pause"]))
		out |= contour(feat, gt["feat"])
	return {k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)) for k, v in out.items()}


# for each key: True if higher is better
HIGHER = dict(register_diff=False, range_diff=False, rate_vs_prompt=False, dur_vs_gt=False, pause_vs_gt=False,
	f0_corr=True, f0_rmse=False, energy_corr=True)
