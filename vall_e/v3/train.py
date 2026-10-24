# Trains the v3 prototype (NAR-demask, classifier or flow head). Logs to <out>/log.jsonl, resumes from <out>/last.pt.
import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import fsq
from .data import BucketSampler, Collator, StoreUtterances, Utterances, add_frames, build_index, pack, to_patches
from .model import V3, flow_loss, flow_sample, metrics, ordinal_loss, snap

NEMO = Path("training/llama-nemo")
GROUPS = [("LibriTTS-R", None), ("Emilia/EN", 20000)]


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


def step_loss(model, b, smoothing, predict=True, stop_weight=0.5, null_p=0.1, stats=None, len_weight=0.1, stop_pos_cap=20.0):
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
	return loss, pred, tgt


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
	ap.add_argument("--accent", action="store_true", help="language (accent) embedding; with --store only")
	ap.add_argument("--lang-drop", type=float, default=0.2, help="share of samples whose language id is dropped to unknown")
	ap.add_argument("--prompt-s", type=float, nargs=2, default=(3.0, 6.0), help="training prompt length range, seconds")
	ap.add_argument("--similar-p", type=float, default=0.0, help="with --store: odds of a prompt from the similar lists")
	ap.add_argument("--noisy-p", type=float, default=0.0, help="with --store: odds of the speaker's noised prompt")
	ap.add_argument("--speaker-alpha", type=float, default=0.0, help="with --store: draw items with weight n_speaker ** -alpha (0: uniform over utterances)")
	ap.add_argument("--store", type=Path, default=None, help="train from a v3 Store (vall_e/v3/store.py) instead of the JSON index")
	ap.add_argument("--groups", nargs="*", default=None, help="with --store: dataset groups to use, e.g. LibriTTS-R Emilia/EN")
	ap.add_argument("--languages", nargs="*", default=None, help="with --store: languages to use, e.g. en")
	ap.add_argument("--max-tokens", type=int, default=0, help="split batches above this many positions (rows x longest row, counting the longest prompt)")
	ap.add_argument("--keep-every", type=int, default=0, help="also keep a copy of the checkpoint every this many steps (step-<n>.pt)")
	ap.add_argument("--init-from", type=Path, default=None, help="start from these weights (new parts keep their init), with a fresh optimizer and schedule")
	args = ap.parse_args()
	args.out.mkdir(parents=True, exist_ok=True)
	device = "cuda"

	if args.store:
		train_ds = StoreUtterances(args.store, "train", crop_prompt=False, groups=args.groups, languages=args.languages,
			prompt_s=tuple(args.prompt_s), similar_p=args.similar_p, noisy_p=args.noisy_p)
		val_ds = StoreUtterances(train_ds.store, "val", seed=1, groups=args.groups, languages=args.languages)
		n_text, vocab, source = train_ds.vocab_size, train_ds.store.m["vocab"], dict(store=str(args.store), groups=args.groups, languages=args.languages)
		prompt_f = train_ds.prompt_f
	else:
		index_path = args.out / "index.json"
		if not index_path.exists():
			t0 = time.time()
			index = build_index(NEMO / "metadata", NEMO / "data", GROUPS)
			index_path.write_text(json.dumps(index))
			print(f"indexed {sum(map(len, index['speakers'].values()))} utterances from {len(index['speakers'])} speakers in {time.time() - t0:.0f}s")
		index = json.loads(index_path.read_text())
		if "frames" not in index:
			t0 = time.time()
			index = add_frames(index)
			index_path.write_text(json.dumps(index))
			print(f"added frame counts in {time.time() - t0:.0f}s")
		train_ds = Utterances(index, "train", crop_prompt=False, prompt_s=tuple(args.prompt_s))
		val_ds = Utterances(index, "val", seed=1)
		n_text, vocab, source = len(index["vocab"]) + 1, index["vocab"], {}
		prompt_f = train_ds.prompt_f
	val_items = [val_ds[i] for i in random.Random(0).sample(range(len(val_ds)), args.val_size)]
	if args.subset:
		# overfit check: train on N fixed utterances (same prompt each time) and evaluate on those same utterances
		full = StoreUtterances(train_ds.store, "train", groups=args.groups, languages=args.languages) if args.store else Utterances(index, "train")
		fixed = [full[i] for i in random.Random(0).sample(range(len(full)), args.subset)]
		train_ds = fixed
		val_items = fixed
	print(f"train {len(train_ds)} utterances, val {len(val_ds)} ({len(val_items)} evaluated)")

	model = V3(n_text, args.d_model, args.layers, args.heads, args.patch, args.rot_frac,
		gated=not args.no_gate, qk_norm=not args.no_qk_norm, head=args.head, ar=args.ar_frac > 0,
		n_lang=train_ds.n_lang if args.accent and args.store else 0, length=args.len_frac > 0).to(device)
	if args.init_from and not (args.out / "last.pt").exists():
		missing, unexpected = model.load_state_dict(torch.load(args.init_from, map_location=device, weights_only=False)["model"], strict=False)
		print(f"initialised from {args.init_from}; new parameters: {missing}; unused: {unexpected}")
	print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
	if args.compile:
		for block in model.blocks:
			block.compile(dynamic=True)  # in place, so state_dict keys are unchanged
	decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "text" not in n]
	no_decay = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "text" not in n)]
	opt = torch.optim.AdamW([dict(params=decay, weight_decay=0.1), dict(params=no_decay, weight_decay=0.0)],
		lr=args.lr, betas=(0.9, 0.95), fused=True)
	step = 0
	if (args.out / "last.pt").exists():
		st = torch.load(args.out / "last.pt", map_location=device, weights_only=False)
		model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); step = st["step"]
		print(f"resumed at step {step}")

	collator = Collator(args.patch, prompt_f, text_drop=args.text_drop, prompt_drop=args.prompt_drop, cond_drop=args.cond_drop, mask_ratio=args.mask_ratio,
		ar_frac=args.ar_frac, len_frac=args.len_frac, lang_drop=args.lang_drop)
	loader_kw = dict(collate_fn=collator, num_workers=args.workers, persistent_workers=True, prefetch_factor=4, pin_memory=True)
	if args.subset:
		sampler = torch.utils.data.RandomSampler(train_ds, replacement=True, num_samples=(args.steps - step) * args.batch)
		dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, sampler=sampler, **loader_kw)
	else:
		# similar-length batches (about 90% real positions instead of about 55% with random batches)
		weights = train_ds.speaker_weights(args.speaker_alpha) if args.speaker_alpha and args.store else None
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
			loss, pred, tgt = step_loss(model, b, args.smoothing, predict=False, stop_weight=args.stop_weight, stats=stats,
				len_weight=args.len_weight * min(1.0, step / max(1, args.len_ramp)), stop_pos_cap=args.stop_pos_cap)
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
			torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, config=model.config, vocab=vocab, **source), args.out / "last.pt.tmp")
			(args.out / "last.pt.tmp").replace(args.out / "last.pt")
			if args.keep_every and step % args.keep_every == 0:
				shutil.copyfile(args.out / "last.pt", args.out / f"step-{step}.pt")


if __name__ == "__main__":
	main()
