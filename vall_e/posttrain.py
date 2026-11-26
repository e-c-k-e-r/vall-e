# Post-training with sampled rewards: generations of the model itself, scored by Whisper (WER / CER) and speaker
# similarity to the prompt.
#
# `length`: GRPO on the length head (as DMOSpeech 2, arXiv 2507.14988, did for its duration predictor): the head's log
# length becomes the mean of a Gaussian policy, each item is generated at `group` lengths drawn from it, and the head
# moves toward the lengths that scored above their group's mean, with a KL pull back to where it started. Only the length
# query and head train. Result (v3-main6-avg3-len, 400 steps): no gain (WER / SIM within noise); the model is at the
# ground truth's WER, and its errors are in the audio, not the length.
#
# `pairs` + `dpo`: preference optimisation of the generator on its own samples, as INTP (arXiv 2505.04113) did for
# MaskGCT / F5-TTS. `pairs` generates `samples` takes per item at the predicted length and scores them; `dpo` trains on
# (better, worse) takes of the same item with the flow-matching form of Diffusion-DPO (arXiv 2311.12908): at a shared
# mask, time and noise, the policy should lower its flow error on the better take more than a frozen reference does,
# relative to the worse one.
#
#   python -m vall_e.posttrain length --init training/v3-main6-avg3-len/last.pt --out training/v3-grpo-len --store <root>
#   python -m vall_e.posttrain pairs --init <ckpt> --out training/dpo-pairs --store <root> --items 3000
#   python -m vall_e.posttrain dpo --init <ckpt> --pairs training/dpo-pairs --out training/v3-dpo
#
# Items come from training speakers (the eval set is validation speakers only), languages weighted roughly as the eval
# set. Scoring needs only the text and a prompt: the item's own recording isn't used, so its transcript quality doesn't
# matter beyond being sensible text.
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from .alloc import expandable_segments

expandable_segments()  # not on HIP 7.15+, where it hangs (vall_e/alloc.py)

import numpy as np
import torch
import torch.nn.functional as F

from . import fsq
from .data import FPS, StoreUtterances, pack, to_patches
from .model import LEN, V3, demask_flow, flow_sample, predict_length, snap

LANG_WEIGHTS = {"en": 0.4, "fr": 0.12, "de": 0.12, "ja": 0.12, "zh": 0.12, "ko": 0.12}


class Scorer:
	"""WER / CER (Whisper large-v3-turbo) and speaker similarity (WavLM + ECAPA) of in-memory 44.1 kHz audio."""
	def __init__(self, device="cuda"):
		from .emb.asr import _pipeline
		# the Whisper pipeline is built before the speaker model (see evalset.score)
		self.asr = _pipeline("openai/whisper-large-v3-turbo", device, "float16")
		from .emb.speaker import _model
		self.spk = _model(device, "float32")
		self.device = device

	@staticmethod
	def _16k(wav):
		import torchaudio
		return torchaudio.functional.resample(wav.float().mean(0, keepdim=True), 44100, 16000)

	@torch.no_grad()
	def embedding(self, wav):
		return self.spk(self._16k(wav).to(self.device))

	def errors(self, wav, text, language):
		"""(errors, units) of the transcript against text."""
		from .emb.text import units, edits
		hyp = self.asr({"raw": self._16k(wav)[0].numpy(), "sampling_rate": 16000}, chunk_length_s=30, batch_size=16,
			generate_kwargs={"task": "transcribe", "language": language})["text"].strip()
		hu, ru = units(hyp, language), units(text, language)
		return edits(hu, ru), len(ru), hyp


class ItemSource:
	"""Training-speaker items for post-training: text in the model's phoneme ids, a 3-10 s prompt from another of the
	speaker's utterances, languages drawn by LANG_WEIGHTS."""
	def __init__(self, store, groups, vocab, model, device, max_s=12.0):
		self.ds = ds = StoreUtterances(store, "train", groups=groups, languages=list(LANG_WEIGHTS), min_s=2.0, max_s=max_s,
			prompt_s=(3.0, 10.0), min_rate=13, min_rate_groups=["LibriVox", "Audiobooks"])
		st = ds.store
		self.own = {c: i + 1 for i, c in enumerate(vocab)}
		self.svocab, self.codes = st.m["vocab"], st.m["languages"]
		langs_of, frames_of = st.language[ds.items], st.frames[ds.items]
		ok = (frames_of >= ds.min_f) & (frames_of <= ds.max_f)
		self.pools = {l: np.nonzero(ok & (langs_of == self.codes.index(l)))[0] for l in LANG_WEIGHTS if l in self.codes}
		self.names = list(self.pools)
		w = np.array([LANG_WEIGHTS[l] for l in self.names])
		self.weights = w / w.sum()
		self.model, self.device = model, device

	def draw(self, rng):
		lang = self.names[rng.choice(len(self.names), p=self.weights)]
		while True:
			i = int(self.pools[lang][rng.integers(len(self.pools[lang]))])
			it = self.ds[i]
			if len(it["prompt"]) >= self.ds.prompt_f[0]:  # speakers with one utterance get no prompt
				break
		text = torch.tensor([self.own[self.svocab[int(t) - 1]] for t in it["text"] if self.svocab[int(t) - 1] in self.own], dtype=torch.long, device=self.device)
		prompt = to_patches(it["prompt"], self.model.patch).to(self.device)
		lid = self.codes.index(lang) + 1 if getattr(self.model, "lang", None) is not None else 0
		return dict(text=text, prompt=prompt, prompt_digits=it["prompt"], lang=lid, language=lang, text_raw=it["text_raw"], index=i)


def decoder(codec, device):
	@torch.no_grad()
	def decode(d):
		c = fsq.digits_to_codes(d).unsqueeze(0).to(device)
		return codec.decode(tokens=c, tokens_len=torch.tensor([c.shape[-1]], device=device))[0].float().cpu()
	return decode


@torch.no_grad()
def demask_batch(model, text, prompt, n_frames, samples, steps=25, cfg=6.0, cfg_prompt=2.0, lang=0, flow_steps=8, sway=-1.0, temperature=1.0):
	"""demask_flow's default path (backbone CFG, optional speaker guidance) for `samples` takes of one text and prompt
	at once, each with its own reveal order and noise. Returns [samples, n_frames, 32] digits."""
	dev, S = text.device, samples
	La = math.ceil(n_frames / model.patch)
	digits = torch.zeros(S, La, model.values, dtype=torch.long, device=dev)
	masked = torch.ones(S, La, dtype=torch.bool, device=dev)
	orders = torch.stack([torch.randperm(La, device=dev) for _ in range(S)])
	texts = [text] * S + [text[:0]] * S + ([text] * S if cfg_prompt > 0 else [])
	prompts = [prompt] * S + [prompt[:0]] * S + ([prompt[:0]] * S if cfg_prompt > 0 else [])
	n_pass = len(texts) // S
	done = 0
	for s in range(steps):
		target = La - int(math.floor(math.cos((s + 1) / steps * math.pi / 2) * La))
		if target == done:
			continue
		rev = orders[:, done:target]  # [S, k]
		h = model.encode(texts, prompts, list(digits) * n_pass, list(masked) * n_pass, langs=[lang] * len(texts))
		h = h.view(n_pass, S, La, -1)
		hr = h.gather(2, rev[None, :, :, None].expand(n_pass, -1, -1, h.shape[-1])).flatten(1, 2)
		vals = flow_sample(model.flow, hr[0], hr[1], cfg=cfg, steps=flow_steps, temperature=temperature, sway=sway,
			h_extra=hr[2] if cfg_prompt > 0 else None, cfg_extra=cfg_prompt)
		digits.scatter_(1, rev[..., None].expand(-1, -1, model.values), snap(vals, model.patch).view(S, -1, model.values))
		masked.scatter_(1, rev, False)
		done = target
	return digits.view(S, -1, fsq.SCALARS)[:, :n_frames]


def load_model(path, device):
	from .export import load_checkpoint
	state, config, vocab = load_checkpoint(path)
	model = V3(**config).eval()
	model.load_state_dict(state)
	return model.to(device), vocab


def log_length(model, text, prompt, lang):
	"""The length head's output (log target length in patches) per item, with gradients: lists of text [Lt] ids and
	prompt [Lp, values] patches."""
	b = pack(text, prompt, [p[:1] for p in prompt], [torch.zeros(1, dtype=torch.bool, device=p.device) for p in prompt],
		modes=[LEN] * len(text), langs=lang)
	h = model.hidden(b, LEN)[b["len_index"]]
	with torch.autocast(h.device.type, enabled=False):  # fp32: see predict_length
		return model.length(h.float()).squeeze(-1)


def grpo_length(args):
	from .evalset import load_codec
	dev = args.device
	model, vocab = load_model(args.init, dev)
	for p in model.parameters():
		p.requires_grad_(False)
	params = [model.len_q, model.length.weight, model.length.bias]
	for p in params:
		p.requires_grad_(True)
	ref = [p.detach().clone() for p in params]
	args.out.mkdir(parents=True, exist_ok=True)
	opt = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.99))
	step = 0
	if (args.out / "last.pt").exists():
		st = torch.load(args.out / "last.pt", map_location="cpu", weights_only=False)
		model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); step = st["step"]
		ref = [st["ref"][k].to(dev) for k in ("len_q", "length.weight", "length.bias")]
		print(f"resumed at step {step}")
	source = ItemSource(args.store, args.groups, vocab, model, dev, args.max_s)
	draw = source.draw
	decode = decoder(load_codec(dev), dev)
	scorer = Scorer(dev)

	@torch.no_grad()
	def rollout(it, n_frames, seed):
		torch.manual_seed(seed)
		with torch.autocast(dev, dtype=torch.bfloat16):
			d = demask_flow(model, it["text"], it["prompt"], n_frames, steps=args.demask_steps, cfg=args.cfg, cfg_prompt=args.cfg_prompt, lang=it["lang"])
		wav = decode(d)
		e, u, hyp = scorer.errors(wav, it["text_raw"], it["language"])
		sim = F.cosine_similarity(scorer.embedding(wav), it["ref_emb"], dim=-1).item()
		wer = min(1.0, e / max(1, u))
		return dict(wer=wer, sim=sim, reward=-wer + args.sim_weight * sim, hyp=hyp)

	def prep(it):
		if "ref_emb" not in it:
			it["ref_emb"] = scorer.embedding(decode(it["prompt_digits"]))
		return it

	def mu_of(items, grad):
		with torch.set_grad_enabled(grad), torch.autocast(dev, dtype=torch.bfloat16):
			return log_length(model, [x["text"] for x in items], [x["prompt"] for x in items], [x["lang"] for x in items])

	def mu_ref(items):
		cur = [p.detach().clone() for p in params]
		with torch.no_grad():
			for p, r in zip(params, ref):
				p.copy_(r)
			m = mu_of(items, False)
			for p, c in zip(params, cur):
				p.copy_(c)
		return m

	# a fixed probe: the same items every time, generated at the head's mean length
	prng = np.random.default_rng(12345)
	probe = [draw(prng) for _ in range(args.probe)]

	def run_probe():
		mus = mu_of(probe, False)
		rs = [rollout(prep(it), max(44, round(math.exp(m) * model.patch)), 777 + k) for k, (it, m) in enumerate(zip(probe, mus.tolist()))]
		return dict(probe_reward=float(np.mean([r["reward"] for r in rs])), probe_wer=float(np.mean([r["wer"] for r in rs])),
			probe_sim=float(np.mean([r["sim"] for r in rs])))

	log = open(args.out / "log.jsonl", "a")
	if step == 0:
		rec = dict(step=0, **run_probe()); log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
	rng = np.random.default_rng(args.seed + step)
	while step < args.steps:
		t0 = time.time()
		items = [prep(draw(rng)) for _ in range(args.items)]
		with torch.no_grad():
			mus = mu_of(items, False)
		groups, stats = [], []  # (item index, sampled log lengths, advantages) for groups with any spread in reward
		for k, (it, m) in enumerate(zip(items, mus.tolist())):
			x = m + args.sigma * np.random.default_rng(int(rng.integers(2**31))).standard_normal(args.group)
			seed = int(rng.integers(2**31))  # the same noise for every length in the group: only the length differs
			rs = [rollout(it, max(44, round(math.exp(v) * model.patch)), seed) for v in x]
			r = np.array([q["reward"] for q in rs])
			stats += rs
			if r.std() < 1e-6:
				continue
			a = (r - r.mean()) / (r.std() + 1e-4)
			groups.append((k, torch.tensor(x, dtype=torch.float, device=dev), torch.tensor(a, dtype=torch.float, device=dev)))
			for q, v, ak in zip(rs, x, a):
				q["dlog"], q["adv"] = float(v - m), float(ak)
		mr = mu_ref(items)  # before the graded pass: mu_ref swaps the parameters in place
		mu = mu_of(items, True)
		# the policy N(mu, sigma) over log length: -A log p(x) = A (x - mu)^2 / (2 sigma^2) + const
		loss_terms = [(a * (x - mu[k]) ** 2).mean() / (2 * args.sigma ** 2) for k, x, a in groups]
		kl = ((mu - mr) ** 2).mean() / (2 * args.sigma ** 2)
		loss = (torch.stack(loss_terms).mean() if loss_terms else torch.zeros((), device=dev)) + args.kl * kl
		opt.zero_grad(set_to_none=True)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(params, 1.0)
		opt.step()
		step += 1
		rec = dict(step=step, loss=loss.item(), kl=kl.item(), groups=len(loss_terms), reward=float(np.mean([q["reward"] for q in stats])),
			wer=float(np.mean([q["wer"] for q in stats])), sim=float(np.mean([q["sim"] for q in stats])),
			shift=float((mu - mr).mean().item()),
			# whether longer or shorter than the mean scored better this step (+: longer)
			corr=float(np.corrcoef([q["dlog"] for q in stats if "dlog" in q], [q["adv"] for q in stats if "adv" in q])[0, 1]) if loss_terms else 0.0,
			s=round(time.time() - t0, 1))
		if step % args.probe_every == 0 or step == args.steps:
			rec.update(run_probe())
		log.write(json.dumps(rec) + "\n"); log.flush()
		print(rec, flush=True)
		if step % args.save_every == 0 or step == args.steps:
			torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, config=model.config, vocab=vocab, source=str(args.init),
				ref=dict(zip(("len_q", "length.weight", "length.bias"), [r.cpu() for r in ref]))), args.out / "last.pt.tmp")
			(args.out / "last.pt.tmp").replace(args.out / "last.pt")
			if args.keep_every and step % args.keep_every == 0:
				import shutil
				shutil.copy(args.out / "last.pt", args.out / f"step-{step}.pt")


def make_pairs(args):
	"""Generates `samples` takes of each of `items` training items at the model's inference settings and predicted
	length, and saves them with their scores in chunks of 100 (resumable): <out>/chunk-XXXX.pt, lists of dict(text
	[Lt] ids, prompt [Lp, 32] and takes [samples, T, 32] uint8 digits, wer, sim, language, lang, text_raw)."""
	from .evalset import load_codec
	dev = args.device
	model, vocab = load_model(args.init, dev)
	source = ItemSource(args.store, args.groups, vocab, model, dev, args.max_s)
	decode = decoder(load_codec(dev), dev)
	scorer = Scorer(dev)
	args.out.mkdir(parents=True, exist_ok=True)
	(args.out / "settings.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
	for c in range(math.ceil(args.items / 100)):
		path = args.out / f"chunk-{c:04d}.pt"
		if path.exists():
			continue
		rng = np.random.default_rng(args.seed * 1_000_003 + c)
		torch.manual_seed(args.seed * 1_000_003 + c)
		rows, t0 = [], time.time()
		for _ in range(min(100, args.items - c * 100)):
			it = source.draw(rng)
			ref = scorer.embedding(decode(it["prompt_digits"]))
			with torch.autocast(dev, dtype=torch.bfloat16):
				n = max(44, predict_length(model, it["text"], it["prompt"], lang=it["lang"]))
				takes = demask_batch(model, it["text"], it["prompt"], n, args.samples, steps=args.demask_steps, cfg=args.cfg,
					cfg_prompt=args.cfg_prompt, lang=it["lang"])
			wers, sims = [], []
			for d in takes:
				wav = decode(d)
				e, u, _ = scorer.errors(wav, it["text_raw"], it["language"])
				wers.append(min(1.0, e / max(1, u)))
				sims.append(F.cosine_similarity(scorer.embedding(wav), ref, dim=-1).item())
			rows.append(dict(text=it["text"].cpu(), prompt=it["prompt_digits"].to(torch.uint8), takes=takes.cpu().to(torch.uint8),
				wer=wers, sim=sims, language=it["language"], lang=it["lang"], text_raw=it["text_raw"], index=it["index"]))
		torch.save(rows, path.with_suffix(".tmp"))
		path.with_suffix(".tmp").replace(path)
		spread = np.mean([max(r["wer"]) - min(r["wer"]) > 0 for r in rows])
		print(f"{path.name}: {len(rows)} items in {time.time() - t0:.0f} s; mean WER {np.mean([np.mean(r['wer']) for r in rows]):.3f}, "
			f"SIM {np.mean([np.mean(r['sim']) for r in rows]):.3f}; items with a WER spread {spread:.0%}", flush=True)


def choose_pair(r, wer_gap, sim_gap, sim_floor):
	"""(better, worse, kind) take indices for one item, or None. "wer": the lowest-WER take (ties: higher SIM) against
	the highest-WER one, if they differ by wer_gap and the better isn't more than sim_floor less similar; "sim": takes
	of equal WER differing in similarity by sim_gap."""
	w, sm = np.array(r["wer"]), np.array(r["sim"])
	order = sorted(range(len(w)), key=lambda k: (w[k], -sm[k]))
	b, worst = order[0], max(range(len(w)), key=lambda k: (w[k], -sm[k]))
	if w[worst] - w[b] >= wer_gap and sm[b] >= sm[worst] - sim_floor:
		return b, worst, "wer"
	same = [k for k in range(len(w)) if w[k] == w[b]]
	lo = min(same, key=lambda k: sm[k])
	if sm[b] - sm[lo] >= sim_gap:
		return b, lo, "sim"
	return None


def dpo(args):
	"""Flow-matching Diffusion-DPO on chosen pairs: per pair one mask (as in training), one flow time and one noise
	draw per repeat, shared by both takes; e = flow MSE over the masked positions, and the loss
	-log sigmoid(-beta (e_pol(w) - e_ref(w) - e_pol(l) + e_ref(l))). The whole model trains; the reference is a frozen
	copy of the initial one."""
	import copy
	dev = args.device
	model, vocab = load_model(args.init, dev)
	ref = copy.deepcopy(model).eval()  # fp32 like the policy: a bf16 copy alone gave a nonzero margin at step 0
	for p in ref.parameters():
		p.requires_grad_(False)
	model.train()
	rows = [r for f in sorted(args.pairs.glob("chunk-*.pt")) for r in torch.load(f, weights_only=False)]
	pairs = []
	for r in rows:
		c = choose_pair(r, args.wer_gap, args.sim_gap, args.sim_floor)
		if c is not None and (c[2] == "wer" or args.sim_pairs):
			pairs.append((r, *c))
	rng = random.Random(args.seed)
	rng.shuffle(pairs)
	n_val = min(max(8, len(pairs) // 20), len(pairs) // 2)
	val, train = pairs[:n_val], pairs[n_val:]
	kinds = {k: sum(p[3] == k for p in pairs) for k in ("wer", "sim")}
	print(f"{len(rows)} items -> {len(pairs)} pairs ({kinds}); {len(train)} train, {len(val)} held out", flush=True)
	args.out.mkdir(parents=True, exist_ok=True)
	opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
	sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
	step = 0
	if (args.out / "last.pt").exists():
		st = torch.load(args.out / "last.pt", map_location="cpu", weights_only=False)
		model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); step = st["step"]
		print(f"resumed at step {step}")

	def errors(net, batch, g):
		"""Per-pair flow MSE of [better, worse], [2, B]: shared masks, times and noise (generator g)."""
		text, prompt, target, masked, x0s, ts = [], [], [], [], [], []
		for r, b, w, _ in batch:
			p = to_patches(r["prompt"].long(), model.patch).to(dev)
			tb, tw = (to_patches(r["takes"][k].long(), model.patch).to(dev) for k in (b, w))
			La = len(tb)
			ratio = math.cos(float(torch.rand((), generator=g)) * math.pi / 2)
			m = torch.zeros(La, dtype=torch.bool, device=dev)
			m[torch.randperm(La, generator=g)[: max(1, round(ratio * La))].to(dev)] = True
			for t in (tb, tw):
				text.append(r["text"].to(dev)); prompt.append(p); target.append(t); masked.append(m)
		with torch.autocast(dev, dtype=torch.bfloat16):
			h = net.encode(text, prompt, target, masked, langs=[r["lang"] for r, *_ in batch for _ in range(2)])
		out, o = [], 0
		for i, t in enumerate(target):
			hi = h[o : o + len(t)][masked[i]]
			o += len(t)
			x1 = fsq.digits_to_values(t[masked[i]]).float()
			if i % 2 == 0:  # the same noise and times for both takes of a pair
				gx = torch.Generator(device="cpu").manual_seed(int(torch.randint(2**31, (1,), generator=g)))
				x0 = torch.randn((args.repeats,) + x1.shape, generator=gx).to(dev)
				tt = torch.sigmoid(torch.randn(args.repeats, len(x1), generator=gx)).to(dev)
			xt = (1 - tt[..., None]) * x0 + tt[..., None] * x1
			with torch.autocast(dev, dtype=torch.bfloat16):
				v = net.flow(xt.flatten(0, 1).to(hi.dtype), tt.flatten(), hi.repeat(args.repeats, 1)).float()
			out.append((v - (x1 - x0).flatten(0, 1)).pow(2).mean())
		return torch.stack(out).view(-1, 2).t()

	def loss_of(batch, seed, grad=True):
		g = torch.Generator().manual_seed(seed)
		with torch.no_grad():
			e_ref = errors(ref, batch, g)
		g = torch.Generator().manual_seed(seed)
		with torch.set_grad_enabled(grad):
			e_pol = errors(model, batch, g)
		margin = (e_pol[0] - e_ref[0]) - (e_pol[1] - e_ref[1])  # negative: the better take gained more
		loss = -F.logsigmoid(-args.beta * margin).mean()
		if args.sft:
			loss = loss + args.sft * e_pol[0].mean()
		return loss, margin

	def validate():
		model.eval()
		ms = []
		for i in range(0, len(val), args.micro):
			_, m = loss_of(val[i : i + args.micro], 99_991 + i, grad=False)
			ms.append(m)
		model.train()
		m = torch.cat(ms)
		return dict(val_acc=float((m < 0).float().mean()), val_margin=float(m.mean()) * 1e3)

	log = open(args.out / "log.jsonl", "a")
	if step == 0:
		rec = dict(step=0, **validate()); log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
	t0 = time.time()
	while step < args.steps:
		batch = [train[(step * args.batch + k) % len(train)] for k in range(args.batch)]
		opt.zero_grad(set_to_none=True)
		losses, margins = [], []
		for i in range(0, len(batch), args.micro):  # gradient accumulation: 8 pairs at once ran out of memory (24 GB)
			part = batch[i : i + args.micro]
			l, m = loss_of(part, args.seed * 7_919 + step * 101 + i)
			(l * len(part) / len(batch)).backward()
			losses.append(l.item() * len(part)); margins.append(m.detach())
		loss, margin = torch.tensor(sum(losses) / len(batch)), torch.cat(margins)
		gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		opt.step(); sched.step()
		step += 1
		if step % 10 == 0:
			rec = dict(step=step, loss=loss.item(), acc=float((margin < 0).float().mean()), margin=float(margin.mean()) * 1e3,
				grad=float(gn), lr=sched.get_last_lr()[0], s_per_step=round((time.time() - t0) / 10, 2))
			t0 = time.time()
			if step % args.eval_every == 0 or step == args.steps:
				rec.update(validate())
			log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
		if step % args.save_every == 0 or step == args.steps:
			torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step, config=model.config,
				vocab=vocab, source=str(args.init)), args.out / "last.pt.tmp")
			(args.out / "last.pt.tmp").replace(args.out / "last.pt")
			if args.keep_every and step % args.keep_every == 0:
				import shutil
				shutil.copy(args.out / "last.pt", args.out / f"step-{step}.pt")


def main():
	ap = argparse.ArgumentParser()
	sub = ap.add_subparsers(dest="cmd", required=True)
	g = sub.add_parser("length", help="GRPO on the length head")
	g.add_argument("--init", type=Path, required=True, help="a checkpoint with a length head")
	g.add_argument("--out", type=Path, required=True)
	g.add_argument("--store", required=True)
	g.add_argument("--groups", nargs="*", default=None)
	g.add_argument("--steps", type=int, default=400)
	g.add_argument("--items", type=int, default=4, help="items per step")
	g.add_argument("--group", type=int, default=8, help="lengths sampled per item")
	g.add_argument("--sigma", type=float, default=0.1, help="policy spread in log length (0.1: about +-10%%)")
	g.add_argument("--sim-weight", type=float, default=1.0, help="reward = -WER (capped at 1) + this * SIM")
	g.add_argument("--kl", type=float, default=0.5, help="pull of the head's mean back to the initial head's")
	g.add_argument("--lr", type=float, default=2e-5, help="Adam over the head's 1024 weights moves the output fast: 2e-4 shifted lengths 8%% in one step")
	g.add_argument("--demask-steps", type=int, default=16, help="rollout demasking steps (16: +0.9 WER vs 25 on the eval set, 1.4x faster)")
	g.add_argument("--cfg", type=float, default=6.0)
	g.add_argument("--cfg-prompt", type=float, default=2.0)
	g.add_argument("--max-s", type=float, default=12.0)
	g.add_argument("--probe", type=int, default=48, help="fixed probe items, generated at the mean length")
	g.add_argument("--probe-every", type=int, default=50)
	g.add_argument("--save-every", type=int, default=25)
	g.add_argument("--keep-every", type=int, default=100)
	g.add_argument("--seed", type=int, default=0)
	g.add_argument("--device", default="cuda")
	g = sub.add_parser("pairs", help="generate and score takes for DPO")
	g.add_argument("--init", type=Path, required=True)
	g.add_argument("--out", type=Path, required=True)
	g.add_argument("--store", required=True)
	g.add_argument("--groups", nargs="*", default=None)
	g.add_argument("--items", type=int, default=3000)
	g.add_argument("--samples", type=int, default=4, help="takes per item")
	g.add_argument("--demask-steps", type=int, default=25)
	g.add_argument("--cfg", type=float, default=6.0)
	g.add_argument("--cfg-prompt", type=float, default=2.0)
	g.add_argument("--max-s", type=float, default=12.0)
	g.add_argument("--seed", type=int, default=0)
	g.add_argument("--device", default="cuda")
	g = sub.add_parser("dpo", help="DPO on scored takes")
	g.add_argument("--init", type=Path, required=True)
	g.add_argument("--pairs", type=Path, required=True, help="a `pairs` output directory")
	g.add_argument("--out", type=Path, required=True)
	g.add_argument("--steps", type=int, default=600)
	g.add_argument("--batch", type=int, default=8, help="pairs per step")
	g.add_argument("--micro", type=int, default=2, help="pairs per forward / backward (gradient accumulation)")
	g.add_argument("--beta", type=float, default=1000.0, help="scale of the flow-error margin (MSE per value) in the sigmoid")
	g.add_argument("--lr", type=float, default=2e-6, help="at beta 1000 margins saturate fast: 2 pairs were fit within 20 steps at ~2e-6")
	g.add_argument("--warmup", type=int, default=50)
	g.add_argument("--repeats", type=int, default=4, help="flow times / noise draws per take")
	g.add_argument("--sft", type=float, default=0.0, help="weight of the plain flow loss on the better take")
	g.add_argument("--wer-gap", type=float, default=0.1)
	g.add_argument("--sim-gap", type=float, default=0.05)
	g.add_argument("--sim-floor", type=float, default=0.03, help="a WER pair's better take may be at most this much less similar")
	g.add_argument("--sim-pairs", action=argparse.BooleanOptionalAction, default=True, help="also pairs of equal WER by similarity")
	g.add_argument("--eval-every", type=int, default=50)
	g.add_argument("--save-every", type=int, default=100)
	g.add_argument("--keep-every", type=int, default=200)
	g.add_argument("--seed", type=int, default=0)
	g.add_argument("--device", default="cuda")
	args = ap.parse_args()
	{"length": grpo_length, "pairs": make_pairs, "dpo": dpo}[args.cmd](args)


if __name__ == "__main__":
	main()
