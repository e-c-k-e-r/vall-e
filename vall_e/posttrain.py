# Post-training with sampled rewards. `length`: GRPO on the length head (as DMOSpeech 2, arXiv 2507.14988, did for its
# duration predictor): the head's log length becomes the mean of a Gaussian policy, each training item is generated at
# `group` lengths drawn from it, every generation is scored (Whisper WER / CER, speaker similarity to the prompt), and
# the head moves toward the lengths that scored above their group's mean, with a KL pull back to where it started. Only
# the length query and head train; the backbone is frozen, so generation at a given length is unchanged.
#
#   python -m vall_e.posttrain length --init training/v3-main6-avg3-len/last.pt --out training/v3-grpo-len --store <root>
#
# Items come from training speakers (the eval set is validation speakers only), languages weighted roughly as the eval
# set. The reward needs only the text and a prompt: the item's own recording isn't used, so its transcript quality
# doesn't matter beyond being sensible text.
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F

from . import fsq
from .data import FPS, StoreUtterances, pack, to_patches
from .model import LEN, V3, demask_flow

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


def log_length(model, text, prompt, lang):
	"""The length head's output (log target length in patches) per item, with gradients: lists of text [Lt] ids and
	prompt [Lp, values] patches."""
	b = pack(text, prompt, [p[:1] for p in prompt], [torch.zeros(1, dtype=torch.bool, device=p.device) for p in prompt],
		modes=[LEN] * len(text), langs=lang)
	h = model.hidden(b, LEN)[b["len_index"]]
	with torch.autocast(h.device.type, enabled=False):  # fp32: see predict_length
		return model.length(h.float()).squeeze(-1)


def grpo_length(args):
	from .export import load_checkpoint
	from .evalset import load_codec
	dev = args.device
	state, config, vocab = load_checkpoint(args.init)
	model = V3(**config).eval()
	model.load_state_dict(state)
	model.to(dev)
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

	ds = StoreUtterances(args.store, "train", groups=args.groups, languages=list(LANG_WEIGHTS), min_s=2.0, max_s=args.max_s,
		prompt_s=(3.0, 10.0), min_rate=13, min_rate_groups=["LibriVox", "Audiobooks"])
	st = ds.store
	own = {c: i + 1 for i, c in enumerate(vocab)}
	svocab = st.m["vocab"]
	codes = st.m["languages"]
	langs_of = st.language[ds.items]
	frames_of = st.frames[ds.items]
	ok = (frames_of >= ds.min_f) & (frames_of <= ds.max_f)
	pools = {l: np.nonzero(ok & (langs_of == codes.index(l)))[0] for l in LANG_WEIGHTS if l in codes}
	names, weights = list(pools), np.array([LANG_WEIGHTS[l] for l in pools])
	weights = weights / weights.sum()
	codec = load_codec(dev)
	scorer = Scorer(dev)

	def draw(rng):
		lang = names[rng.choice(len(names), p=weights)]
		i = int(pools[lang][rng.integers(len(pools[lang]))])
		it = ds[i]
		text = torch.tensor([own[svocab[int(t) - 1]] for t in it["text"] if svocab[int(t) - 1] in own], dtype=torch.long, device=dev)
		prompt = to_patches(it["prompt"], model.patch).to(dev)
		lid = codes.index(lang) + 1 if getattr(model, "lang", None) is not None else 0
		return dict(text=text, prompt=prompt, prompt_digits=it["prompt"], lang=lid, language=lang, text_raw=it["text_raw"])

	@torch.no_grad()
	def decode(d):
		c = fsq.digits_to_codes(d).unsqueeze(0).to(dev)
		return codec.decode(tokens=c, tokens_len=torch.tensor([c.shape[-1]], device=dev))[0].float().cpu()

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
	args = ap.parse_args()
	if args.cmd == "length":
		grpo_length(args)


if __name__ == "__main__":
	main()
