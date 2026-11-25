# Training (docs/train.md): masked generation with the flow head on a data store, plus the optional AR, LEN and CTC
# objectives and frozen-backbone heads (--train-only). Logs to <out>/log.jsonl, resumes from <out>/last.pt.
import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import fsq
from .data import FPS, BucketSampler, Collator, StoreUtterances, pack, to_patches
from .model import V3, flow_loss, flow_sample, metrics, ordinal_loss, snap



def lr_scale(step, warmup, steps, decay_frac):
	"""Warmup, constant, then linear decay to zero over the last decay_frac of steps (WSD)."""
	if step < warmup:
		return (step + 1) / warmup
	decay_start = steps * (1 - decay_frac)
	return 1.0 if step < decay_start else max(0.0, (steps - step) / (steps - decay_start))


def make_batch(items, patch, device, rng, cond_drop=0.1, mask_ratio=None):
	"""Masks each target with ratio cos(u * pi / 2) (or a fixed ratio) and drops both conditions for cond_drop of samples.
	Returns text, prompt, target, masked lists and the language ids."""
	text, prompt, target, masked, langs = [], [], [], [], []
	for it in items:
		a = to_patches(it["target"], patch).to(device)
		r = math.cos(rng.random() * math.pi / 2) if mask_ratio is None else mask_ratio
		n = max(1, round(r * len(a)))
		m = torch.zeros(len(a), dtype=torch.bool)
		m[torch.tensor(rng.sample(range(len(a)), n))] = True
		drop = rng.random() < cond_drop
		text.append(it["text"][:0].to(device) if drop else it["text"].to(device))
		p = to_patches(it["prompt"], patch).to(device)
		prompt.append(p[:0] if drop else p)
		target.append(a); masked.append(m.to(device)); langs.append(it.get("lang", 0))
	return text, prompt, target, masked, langs


def step_loss(model, b, smoothing, predict=True, stop_weight=0.5, null_p=0.1, stats=None, len_weight=0.1, stop_pos_cap=20.0, ctc_weight=0.0):
	"""b: data.pack() tensors on the model's device. Returns the loss, predicted level indices of the masked positions
	(None for the flow head unless predict, as it has to be sampled) and their targets. For AR-capable models, the flow
	head sees each predicted patch's previous patch, its condition is replaced by the null embedding for null_p of
	positions (for head-only CFG), and AR rows add a stop loss; LEN rows add len_weight times the L1 error of the log
	length. The parts go into `stats` if given."""
	loss, pred, tgt = _main_loss(model, b, smoothing, predict, stop_weight, null_p, stats, stop_pos_cap)
	h = b.pop("_h")
	if getattr(model, "len_q", None) is not None and len(b["len_index"]):
		err = (model.length(h[b["len_index"]]).squeeze(-1).float() - b["len_tgt"].clamp(min=1).log()).abs().mean()
		loss = loss + len_weight * err
		if stats is not None:
			stats["len_loss"] = err.item()
	if getattr(model, "ctc", None) is not None and ctc_weight > 0:
		c = ctc_loss(model, b, h)
		if c is not None:
			loss = loss + ctc_weight * c
			if stats is not None:
				stats["ctc_loss"] = c.item()
	return loss, pred, tgt


ROW_KEYS = ("text_ids", "digits", "kind", "masked", "lang", "mode", "prefix_len")


def distill_loss(model, teacher, b, cfg, cfg_prompt, repeats=4):
	"""Guidance distillation: on the full-condition rows, the student's flow velocity matches the teacher's guided one,
	v_null + cfg (v_cond - v_null) + cfg_prompt (v_cond - v_noprompt), at the same noisy values and times, so sampling
	without guidance (one backbone pass per step instead of three) gives what guided sampling gave. b: a Collator(distill=True)
	batch, [cond rows | null rows | promptless rows]."""
	B = int(b["n_cond"])
	n1 = int((b["kind"][:B] == 3).sum())  # target positions of the cond rows: the first n1 of tgt_index
	m1 = b["tgt_masked"][:n1]
	with torch.no_grad():
		hc, hn, hp = teacher.hidden(b)[b["tgt_index"]][b["tgt_masked"]].chunk(3)
	hs = model.hidden({k: b[k][:B] for k in ROW_KEYS})[b["tgt_index"][:n1]][m1]
	x1 = fsq.digits_to_values(b["tgt"].long()[:n1][m1]).repeat(repeats, 1)
	x0 = torch.randn_like(x1)
	t = torch.sigmoid(torch.randn(len(x1), device=x1.device))
	xt = ((1 - t[:, None]) * x0 + t[:, None] * x1).to(hs.dtype)
	with torch.no_grad():
		R = lambda h: h.repeat(repeats, 1)
		vc, vn, vp = teacher.flow(torch.cat([xt] * 3), torch.cat([t] * 3), torch.cat([R(hc), R(hn), R(hp)])).float().chunk(3)
		target = vn + cfg * (vc - vn) + cfg_prompt * (vc - vp)
	return F.mse_loss(model.flow(xt, t, R(hs)).float(), target)


class Repa:
	"""REPA (arXiv 2410.06940; for TTS, AG-REPA 2603.01006): a shallow block's output at the target positions, through
	a small training-only projector, is pulled (cosine) toward frozen speech-model features of the target audio
	(vall_e/repa.py bakes them for a fixed subset; the batch carries them as repa_tgt, in tgt_index order). The early
	layers then carry the speech model's phonetic structure, meant to speed up and firm up learning the alignment."""
	def __init__(self, model, dim, layer=3, device="cuda"):
		self.tap = {}
		model.blocks[layer].register_forward_hook(lambda m, i, o: self.tap.__setitem__("h", o))
		d = model.config["d_model"]
		self.proj = torch.nn.Sequential(torch.nn.Linear(d, 2048), torch.nn.SiLU(), torch.nn.Linear(2048, 2048), torch.nn.SiLU(),
			torch.nn.Linear(2048, dim)).to(device)

	def loss(self, b):
		z = self.tap["h"].flatten(0, 1)[b["tgt_index"]].float()
		return 1 - F.cosine_similarity(self.proj(z), b["repa_tgt"].float(), dim=-1).mean()


def ctc_loss(model, b, h):
	"""CTC of each NAR row's phonemes over its target positions (all of them, masked or not), or None if no row has
	both text and a target."""
	kind, B, L = b["kind"], b["kind"].shape[0], b["kind"].shape[1]
	rows = [i for i in range(B) if int(b["mode"][i]) == 0 and bool((kind[i] == 1).any()) and bool((kind[i] == 3).any())]
	if not rows:
		return None
	pos = [i * L + (kind[i] == 3).nonzero().squeeze(-1) for i in rows]
	tgts = [b["text_ids"][i][kind[i] == 1] for i in rows]
	T = max(len(p) for p in pos)
	idx = torch.stack([F.pad(p, (0, T - len(p)), value=int(p[-1])) for p in pos], 1)  # [T, N]
	logp = F.log_softmax(model.ctc(h[idx]).float(), -1)
	return F.ctc_loss(logp, torch.cat(tgts), torch.tensor([len(p) for p in pos]), torch.tensor([len(t) for t in tgts]),
		blank=0, zero_infinity=True)


def _main_loss(model, b, smoothing, predict, stop_weight, null_p, stats, stop_pos_cap):
	m = b["tgt_masked"]
	tgt = b["tgt"].long()[m]
	h_all = b["_h"] = model.hidden(b)
	if not m.any():  # a batch of LEN rows only: nothing to generate (a mean over no positions would be NaN)
		return h_all.sum() * 0.0, None, tgt
	h = h_all[b["tgt_index"]][m]
	if model.head_type != "flow":
		logits = model.logits(h)
		return ordinal_loss(logits, tgt, smoothing), logits.argmax(-1), tgt
	if not model.ar:
		pred = snap(flow_sample(model.flow, h), model.patch) if predict else None
		return flow_loss(model.flow, h, tgt), pred, tgt
	hist, has = b["hist"].long()[m], b["has_hist"][m]
	pred = snap(flow_sample(model.flow, h, hist=fsq.digits_to_values(hist), has_hist=has), model.patch) if predict else None
	if model.training and null_p > 0:
		drop = torch.rand(len(h), device=h.device) < null_p
		h = torch.where(drop[:, None], model.null_h.to(h.dtype), h)
	loss = flow_loss(model.flow, h, tgt, hist=hist, has_hist=has)
	if len(b["stop_index"]):
		logit = model.stop(h_all[b["stop_index"]]).squeeze(-1).float()
		y = b["stop_tgt"].float()
		# one positive per AR utterance: weight it by the negatives per positive so both classes count equally, capped
		# (uncapped, a long batch's weight of several hundred spiked the gradient norm)
		pos_weight = ((1 - y).sum() / y.sum().clamp(min=1)).clamp(max=stop_pos_cap).detach()
		stop = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)
		loss = loss + stop_weight * stop
		if stats is not None:
			stats["stop_loss"] = stop.item()
	return loss, pred, tgt


@torch.no_grad()
def evaluate(model, val_items, patch, device, smoothing, ratios=(0.5, 0.8)):
	model.eval()
	res = {}
	if getattr(model, "ar", False):
		res |= evaluate_ar(model, val_items, patch, device)
	if getattr(model, "len_q", None) is not None:
		res |= evaluate_len(model, val_items, patch, device)
	for r in ratios:
		rng = random.Random(1234)
		agg, n = {}, 0
		for i in range(0, len(val_items), 16):
			with torch.autocast("cuda", dtype=torch.bfloat16):
				t, p, a, m, langs = make_batch(val_items[i : i + 16], patch, device, rng, cond_drop=0, mask_ratio=r)
				loss, pred, tgt = step_loss(model, pack(t, p, a, m, langs=langs), smoothing)
			for k, v in dict(loss=loss.item(), **metrics(pred, tgt)).items():
				agg[k] = agg.get(k, 0) + v * len(tgt)
			n += len(tgt)
		res |= {f"val@{r}/{k}": v / n for k, v in agg.items()}
	res |= evaluate_text_use(model, val_items, patch, device, smoothing)
	model.train()
	return res


@torch.no_grad()
def evaluate_text_use(model, val_items, patch, device, smoothing, ratio=0.8):
	"""How much the model reads the text: the relative loss increase when each item gets another item's text, at the
	same masks and flow noise. Teacher-forced losses barely show a model that ignores the text (v3-main sampled at 59%
	WER with losses on par with v3-flow-large); this does (v3-flow-large +5.1%, v3-main4 +0.7%)."""
	rng = random.Random(1234)
	right = wrong = 0.0
	for i in range(0, len(val_items), 16):
		t, p, a, m, langs = make_batch(val_items[i : i + 16], patch, device, rng, cond_drop=0, mask_ratio=ratio)
		for texts, acc in ((t, "right"), (t[1:] + t[:1], "wrong")):
			with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == "cuda" else []):
				torch.manual_seed(i)
				with torch.autocast("cuda", dtype=torch.bfloat16):
					loss = step_loss(model, pack(texts, p, a, m, langs=langs), smoothing, predict=False)[0].item()
			if acc == "right":
				right += loss
			else:
				wrong += loss
	return {"val/text_use": (wrong - right) / right}


@torch.no_grad()
def evaluate_ar(model, val_items, patch, device):
	"""Teacher-forced AR: flow loss and sampled-patch metrics, and how far the most likely stop position is from the true
	end (in patches)."""
	agg, n, stop_err, n_utt = {}, 0, 0.0, 0
	rng = random.Random(1234)
	for i in range(0, len(val_items), 16):
		text, prompt, target, masked, langs = make_batch(val_items[i : i + 16], patch, device, rng, cond_drop=0)
		b = pack(text, prompt, target, masked, modes=[1] * len(text), langs=langs)
		with torch.autocast("cuda", dtype=torch.bfloat16):
			loss, pred, tgt = step_loss(model, b, 0.0)
			logit = model.stop(model.hidden(b)[b["stop_index"]]).squeeze(-1).float()
		for k, v in dict(loss=loss.item(), **metrics(pred, tgt)).items():
			agg[k] = agg.get(k, 0) + v * len(tgt)
		n += len(tgt)
		start = 0
		for a in target:
			seg = logit[start : start + len(a) + 1]
			stop_err += abs(int(seg.argmax()) - len(a)); n_utt += 1
			start += len(a) + 1
	return {f"val@ar/{k}": v / n for k, v in agg.items()} | {"val@ar/stop_pos_err": stop_err / n_utt}


@torch.no_grad()
def evaluate_len(model, val_items, patch, device):
	"""LEN mode: mean absolute error of the predicted length, as a fraction of the true length."""
	rng, errs = random.Random(1234), []
	for i in range(0, len(val_items), 16):
		text, prompt, target, masked, langs = make_batch(val_items[i : i + 16], patch, device, rng, cond_drop=0)
		b = pack(text, prompt, target, masked, modes=[2] * len(text), langs=langs)
		with torch.autocast("cuda", dtype=torch.bfloat16):
			pred = model.length(model.hidden(b)[b["len_index"]]).squeeze(-1).float().exp()
		errs.append(((pred - b["len_tgt"]).abs() / b["len_tgt"]).cpu())
	return {"val@len/rel_err": torch.cat(errs).mean().item()}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--steps", type=int, default=100_000)
	ap.add_argument("--batch", type=int, default=32)
	ap.add_argument("--lr", type=float, default=5e-4)
	ap.add_argument("--warmup", type=int, default=1000)
	ap.add_argument("--decay-frac", type=float, default=0.2)
	ap.add_argument("--d-model", type=int, default=512)
	ap.add_argument("--layers", type=int, default=12)
	ap.add_argument("--heads", type=int, default=8)
	ap.add_argument("--patch", type=int, default=2)
	ap.add_argument("--rot-frac", type=float, default=1.0)
	ap.add_argument("--no-gate", action="store_true")
	ap.add_argument("--no-qk-norm", action="store_true")
	ap.add_argument("--smoothing", type=float, default=0.1)
	ap.add_argument("--head", choices=["classifier", "flow"], default="classifier")
	ap.add_argument("--workers", type=int, default=8)
	ap.add_argument("--eval-every", type=int, default=2000)
	ap.add_argument("--val-size", type=int, default=256)
	ap.add_argument("--subset", type=int, default=0, help="train and evaluate on this many fixed utterances (overfit check)")
	ap.add_argument("--compile", action="store_true", help="torch.compile each transformer block")
	ap.add_argument("--grad-ckpt", action="store_true", help="activation checkpointing per transformer block (bigger models in 24 GB, ~30%% slower)")
	# fixed 0.8 from scratch (v3-main to step 124k) never trains generation without audio anchors: teacher-forced
	# metrics matched v3-flow-large, but sampled WER was 59% against its 4%
	ap.add_argument("--mask-ratio", type=lambda v: None if v == "cos" else float(v), default=None,
		help="training mask ratio: `cos` (default) draws cos(u * pi / 2) per sample; a float fixes it (AutoNAT)")
	ap.add_argument("--ar-frac", type=float, default=0.0, help="share of samples trained in AR mode; > 0 builds the AR parts")
	ap.add_argument("--stop-weight", type=float, default=0.5)
	ap.add_argument("--stop-pos-cap", type=float, default=20.0, help="cap on the stop loss's positive weight")
	ap.add_argument("--len-frac", type=float, default=0.0, help="share of samples trained in LEN mode; > 0 builds the length head")
	ap.add_argument("--len-weight", type=float, default=0.1)
	ap.add_argument("--len-ramp", type=int, default=10000, help="steps over which the length loss weight ramps up from 0")
	ap.add_argument("--cond-drop", type=float, default=0.0, help="share of samples with both text and prompt dropped (CFG's null)")
	ap.add_argument("--text-drop", type=float, default=0.1, help="share of samples trained without text (independent of the prompt's)")
	ap.add_argument("--prompt-drop", type=float, default=0.1, help="share of samples trained without a prompt")
	ap.add_argument("--accent", action="store_true", help="language (accent) embedding")
	ap.add_argument("--lang-drop", type=float, default=0.2, help="share of samples whose language id is dropped to unknown")
	ap.add_argument("--prompt-s", type=float, nargs=2, default=(3.0, 6.0), help="training prompt length range, seconds")
	ap.add_argument("--similar-p", type=float, default=0.0, help="odds of a prompt from the similar lists")
	ap.add_argument("--noisy-p", type=float, default=0.0, help="odds of the speaker's noised prompt")
	ap.add_argument("--ctc-weight", type=float, default=0.0, help="auxiliary phoneme-recognition (CTC) loss weight; > 0 builds the head")
	ap.add_argument("--n-modes", type=int, default=None, help="mode embedding size (default 5 with --ns-p / --sr-p, else 3)")
	ap.add_argument("--span-p", type=float, default=0.0, help="share of targets masked as one contiguous span (editing / continuation)")
	ap.add_argument("--ns-p", type=float, default=0.0, help="share of items that are noise suppression (needs noisy-targets in the store)")
	ap.add_argument("--sr-p", type=float, default=0.0, help="share of items that are speech removal")
	ap.add_argument("--pad-p", type=float, default=0.0, help="share of targets padded with trailing silence")
	ap.add_argument("--cont-p", type=float, default=0.0,
		help="share of items made continuations (tts-c): another utterance of the speaker known in front, only the target generated")
	ap.add_argument("--cont-s", type=float, default=8.0, help="longest utterance put in front of a continuation (s)")
	ap.add_argument("--min-rate", type=float, default=None, help="drop items under this many phonemes / s (misaligned segments)")
	ap.add_argument("--min-rate-groups", nargs="*", default=None, help="groups --min-rate applies to (default all)")
	ap.add_argument("--repa-weight", type=float, default=0.0, help="REPA alignment loss weight (see Repa); > 0 enables it")
	ap.add_argument("--repa-layer", type=int, default=3, help="the block whose output is aligned")
	ap.add_argument("--repa-feats", type=Path, default=None, help="baked REPA targets (vall_e/repa.py bake); training uses only their utterances")
	ap.add_argument("--items-from", type=Path, default=None, help="train only on these store utterances (an ids.npy, e.g. a REPA bake's, for paired ablations)")
	ap.add_argument("--text-prenet", type=int, default=0, help="ConvNeXt blocks over the text embeddings")
	ap.add_argument("--distill-from", type=Path, default=None, help="guidance distillation from this teacher (see distill_loss)")
	ap.add_argument("--distill-cfg", type=float, default=6.0)
	ap.add_argument("--distill-cfg-prompt", type=float, default=2.0)
	ap.add_argument("--extra-store", type=Path, nargs="*", default=None, help="more stores to train on (e.g. an ingest.py store; same phoneme vocabulary)")
	ap.add_argument("--holdout-speakers", type=Path, default=None, help='JSON list of "group/speaker" never trained on (a zero-shot test set)')
	ap.add_argument("--oversample-groups", nargs="*", default=None, help="groups to draw as --oversample-p of all draws")
	ap.add_argument("--oversample-p", type=float, default=0.3)
	ap.add_argument("--min-s", type=float, default=2.0, help="shortest target, seconds")
	ap.add_argument("--max-s", type=float, default=12.0, help="longest target, seconds")
	ap.add_argument("--long-p", type=float, default=0.0, help="share of draws from targets over --long-s (0: as they come)")
	ap.add_argument("--long-s", type=float, default=12.0)
	ap.add_argument("--speaker-alpha", type=float, default=0.0, help="draw items with weight n_speaker ** -alpha (0: uniform over utterances)")
	ap.add_argument("--store", type=Path, required=True, help="the data store (store.py)")
	ap.add_argument("--groups", nargs="*", default=None, help="dataset groups to use, e.g. LibriTTS-R Emilia/EN")
	ap.add_argument("--languages", nargs="*", default=None, help="languages to use, e.g. en")
	ap.add_argument("--max-tokens", type=int, default=0, help="split batches above this many positions (rows x longest row, counting the longest prompt)")
	ap.add_argument("--keep-every", type=int, default=0, help="also keep a copy of the checkpoint every this many steps (step-<n>.pt)")
	ap.add_argument("--train-only", nargs="*", default=None,
		help="train only parameters whose names start with these prefixes (e.g. len_q length), the rest frozen")
	ap.add_argument("--init-from", type=Path, default=None, help="start from these weights (new parts keep their init), with a fresh optimizer and schedule")
	args = ap.parse_args()
	args.out.mkdir(parents=True, exist_ok=True)
	device = "cuda"

	train_ds = StoreUtterances(args.store, "train", crop_prompt=False, groups=args.groups, languages=args.languages,
		min_s=args.min_s, max_s=args.max_s, prompt_s=tuple(args.prompt_s), similar_p=args.similar_p, noisy_p=args.noisy_p, min_rate=args.min_rate,
		min_rate_groups=args.min_rate_groups, pad_p=args.pad_p, ns_p=args.ns_p, sr_p=args.sr_p, cont_p=args.cont_p, cont_s=args.cont_s,
		only=np.load(args.items_from or args.repa_feats / "ids.npy") if (args.items_from or args.repa_feats) else None, repa=args.repa_feats,
		exclude_speakers=json.loads(args.holdout_speakers.read_text()) if args.holdout_speakers else None)
	if args.extra_store:
		# more stores (ingest.py), with the same length / rate filters and held-out speakers
		from .data import ConcatUtterances
		extra = [StoreUtterances(x, "train", crop_prompt=False, min_s=args.min_s, max_s=args.max_s, prompt_s=tuple(args.prompt_s), cont_p=args.cont_p, cont_s=args.cont_s,
			exclude_speakers=json.loads(args.holdout_speakers.read_text()) if args.holdout_speakers else None) for x in args.extra_store]
		print(f"extra stores: {[(str(x), len(e)) for x, e in zip(args.extra_store, extra)]}")
		train_ds = ConcatUtterances([train_ds] + extra)
	if (args.ns_p or args.sr_p) and train_ds.pairs is None:
		raise SystemExit("--ns-p / --sr-p need <store>/noisy-targets (python -m vall_e.noisy bake-targets)")
	val_ds = StoreUtterances(train_ds.store, "val", seed=1, groups=args.groups, languages=args.languages)
	n_text, vocab, source = train_ds.vocab_size, train_ds.store.m["vocab"], dict(store=str(args.store), groups=args.groups, languages=args.languages)
	prompt_f = train_ds.prompt_f
	val_items = [val_ds[i] for i in random.Random(0).sample(range(len(val_ds)), args.val_size)]
	if args.subset:
		# overfit check: train on N fixed utterances (same prompt each time) and evaluate on those same utterances
		full = StoreUtterances(train_ds.store, "train", groups=args.groups, languages=args.languages)
		fixed = [full[i] for i in random.Random(0).sample(range(len(full)), args.subset)]
		train_ds = fixed
		val_items = fixed
	print(f"train {len(train_ds)} utterances, val {len(val_ds)} ({len(val_items)} evaluated)")

	model = V3(n_text, args.d_model, args.layers, args.heads, args.patch, args.rot_frac,
		gated=not args.no_gate, qk_norm=not args.no_qk_norm, head=args.head, ar=args.ar_frac > 0,
		n_lang=train_ds.n_lang if args.accent else 0, length=args.len_frac > 0, ctc=args.ctc_weight > 0,
		n_modes=args.n_modes or (5 if (args.ns_p or args.sr_p) else 3), text_prenet=args.text_prenet).to(device)
	if args.init_from and not (args.out / "last.pt").exists():
		init = torch.load(args.init_from, map_location=device, weights_only=False)["model"]
		own = model.state_dict()
		for k, v in list(init.items()):
			if k in own and own[k].shape != v.shape and k in ("mode.weight", "text.weight") and v.shape[0] < own[k].shape[0]:
				# more modes / phonemes (an extended vocabulary) than the checkpoint: its rows carry over, the new ones keep
				# their initialisation
				grown = own[k].clone(); grown[: v.shape[0]] = v; init[k] = grown
				print(f"{k}: {v.shape[0]} -> {own[k].shape[0]} rows")
		missing, unexpected = model.load_state_dict(init, strict=False)
		print(f"initialised from {args.init_from}; new parameters: {missing}; unused: {unexpected}")
	print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
	model.grad_ckpt = args.grad_ckpt
	if args.compile:
		for block in model.blocks:
			block.compile(dynamic=True)  # in place, so state_dict keys are unchanged
	if args.train_only:
		# e.g. a length head on a frozen model: generation can't change
		for n, p in model.named_parameters():
			p.requires_grad_(any(n.startswith(x) for x in args.train_only))
		print(f"training only {[n for n, p in model.named_parameters() if p.requires_grad]}")
	params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
	decay = [p for n, p in params if p.ndim >= 2 and "text" not in n]
	no_decay = [p for n, p in params if not (p.ndim >= 2 and "text" not in n)]
	groups = [dict(params=decay, weight_decay=0.1), dict(params=no_decay, weight_decay=0.0)]
	repa = None
	if args.repa_weight:
		repa = Repa(model, train_ds.repa["feats"].shape[1], args.repa_layer, device)
		groups.append(dict(params=list(repa.proj.parameters()), weight_decay=0.0))
		print(f"REPA: block {args.repa_layer} -> {args.repa_feats}, weight {args.repa_weight}")
	opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), fused=True)
	step = 0
	if (args.out / "last.pt").exists():
		st = torch.load(args.out / "last.pt", map_location=device, weights_only=False)
		model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); step = st["step"]
		if repa is not None and "repa" in st:
			repa.proj.load_state_dict(st["repa"])
		print(f"resumed at step {step}")

	teacher = None
	if args.distill_from:
		from .export import load_checkpoint
		tstate, tconfig, _ = load_checkpoint(args.distill_from)
		teacher = V3(**tconfig).eval()
		teacher.load_state_dict(tstate)
		teacher.to(device)
		for p in teacher.parameters():
			p.requires_grad_(False)
		args.text_drop = args.prompt_drop = args.cond_drop = 0.0  # the collator adds the unconditional rows itself
		print(f"distilling guidance (cfg {args.distill_cfg}, speaker {args.distill_cfg_prompt}) from {args.distill_from}")
	collator = Collator(args.patch, prompt_f, text_drop=args.text_drop, prompt_drop=args.prompt_drop, cond_drop=args.cond_drop, span_p=args.span_p, mask_ratio=args.mask_ratio,
		ar_frac=args.ar_frac, len_frac=args.len_frac, lang_drop=args.lang_drop, distill=teacher is not None)
	loader_kw = dict(collate_fn=collator, num_workers=args.workers, persistent_workers=True, prefetch_factor=4, pin_memory=True)
	if args.subset:
		sampler = torch.utils.data.RandomSampler(train_ds, replacement=True, num_samples=(args.steps - step) * args.batch)
		dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, sampler=sampler, **loader_kw)
	else:
		# similar-length batches (about 90% real positions instead of about 55% with random batches)
		weights = train_ds.speaker_weights(args.speaker_alpha) if args.speaker_alpha else None
		if args.oversample_groups:
			# the named groups drawn as --oversample-p of all draws (e.g. game audio, under 1% of the store)
			names = train_ds.item_groups()
			hit = {g for g in set(names) if any(g == x or g.startswith(x + "/") for x in args.oversample_groups)}
			inside = np.array([g in hit for g in names])
			gids = sorted(hit)
			w = np.ones(len(inside)) if weights is None else np.asarray(weights, dtype=np.float64)
			weights = np.where(inside, w * args.oversample_p / w[inside].sum(), w * (1 - args.oversample_p) / w[~inside].sum())
			print(f"oversampling {len(gids)} groups ({int(inside.sum())} utterances) to {args.oversample_p:.0%} of draws")
		if args.long_p:
			# utterances over --long-s drawn as --long-p of all draws (they're 7% of the store over 12 s)
			long = np.asarray(train_ds.frames) > args.long_s * FPS
			w = np.ones(len(long)) if weights is None else np.asarray(weights, dtype=np.float64)
			ok = (np.asarray(train_ds.frames) >= train_ds.min_f) & (np.asarray(train_ds.frames) <= train_ds.max_f)
			wl, ws = w[long & ok].sum(), w[~long & ok].sum()
			weights = np.where(long, w * args.long_p / wl, w * (1 - args.long_p) / ws)
		dl = torch.utils.data.DataLoader(train_ds, batch_sampler=BucketSampler(train_ds, args.batch, args.steps - step, args.patch,
			seed=step, weights=weights, max_tokens=args.max_tokens or None, extra=math.ceil(prompt_f[1] / args.patch) + 1), **loader_kw)
	log = open(args.out / "log.jsonl", "a")
	model.train()
	t0, run = time.time(), {}
	for b in dl:
		if step >= args.steps:
			break
		for g in opt.param_groups:
			g["lr"] = args.lr * lr_scale(step, args.warmup, args.steps, args.decay_frac)
		with torch.autocast("cuda", dtype=torch.bfloat16):
			b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
			stats = {}
			if teacher is not None:
				loss, pred, tgt = distill_loss(model, teacher, b, args.distill_cfg, args.distill_cfg_prompt), None, None
			else:
				loss, pred, tgt = step_loss(model, b, args.smoothing, predict=False, stop_weight=args.stop_weight, stats=stats,
					len_weight=args.len_weight * min(1.0, step / max(1, args.len_ramp)), stop_pos_cap=args.stop_pos_cap, ctc_weight=args.ctc_weight)
				if repa is not None:
					rl = repa.loss(b)
					loss = loss + args.repa_weight * rl
					stats["repa_loss"] = rl.item()
		opt.zero_grad(set_to_none=True)
		loss.backward()
		gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		opt.step()
		step += 1
		if step % 50 == 0:
			for k, v in dict(loss=loss.item(), gnorm=gnorm.item(), **stats, **(metrics(pred, tgt) if pred is not None else {})).items():
				run[k] = run.get(k, 0) + v
		if step % 500 == 0:
			rec = {k: v / 10 for k, v in run.items()} | dict(step=step, lr=opt.param_groups[0]["lr"], sec_per_step=(time.time() - t0) / 500)
			log.write(json.dumps(rec) + "\n"); log.flush()
			print(" ".join(f"{k} {v:.4g}" for k, v in rec.items()), flush=True)
			t0, run = time.time(), {}
		if step % args.eval_every == 0 or step == args.steps:
			res = evaluate(model, val_items, args.patch, device, args.smoothing) | dict(step=step)
			log.write(json.dumps(res) + "\n"); log.flush()
			print("[eval] " + " ".join(f"{k} {v:.4g}" for k, v in res.items()), flush=True)
			torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, config=model.config, vocab=vocab, **source,
				**({"repa": repa.proj.state_dict()} if repa is not None else {})), args.out / "last.pt.tmp")
			(args.out / "last.pt.tmp").replace(args.out / "last.pt")
			if args.keep_every and step % args.keep_every == 0:
				shutil.copyfile(args.out / "last.pt", args.out / f"step-{step}.pt")


if __name__ == "__main__":
	main()
