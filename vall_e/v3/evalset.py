# A fixed, larger evaluation set for v3 checkpoints, with confidence intervals. The earlier checks (6 items per language,
# 16 English) disagreed with each other by more than the differences being judged; this one freezes its inputs so runs
# are paired, and reports bootstrap intervals so a difference can be told from noise.
#
#   python -m vall_e.v3.evalset build --store <root> --out evalsets/v3-eval.json            # once
#   python -m vall_e.v3.evalset run --set evalsets/v3-eval.json --ckpt <ckpt> --out <dir> [--length predict --length-scale 1.2]
#   python -m vall_e.v3.evalset compare <dir A> <dir B>
#
# Items: validation speakers only (never trained on), English 100 + 30 per other language by default, each with a fixed
# prompt (another utterance of the same speaker, a fixed crop). Scoring: Whisper large-v3-turbo in the item's language,
# WER (CER for zh / ja / ko) as in sample.py, pooled over the units of each language.
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .data import FPS, _dup_key, to_patches

CER_LANGUAGES = ("zh", "ja", "ko")


# English is stratified over its sources (drawn uniformly it was two-thirds Audiobooks, whose own transcripts are the
# least reliable); the other languages are all Emilia
STRATA = {"en": {"LibriTTS-R": 25, "LibriVox": 25, "Audiobooks": 25, "Emilia/EN": 25},
	"fr": {"Emilia/FR": 30}, "de": {"Emilia/DE": 30}, "ja": {"Emilia/JA": 30}, "zh": {"Emilia/ZH": 30}, "ko": {"Emilia/KO": 30}}


def build(store, out, strata=None, groups=None, prompt_s=6.0, seed=0, min_s=2.0, max_s=12.0, per_speaker=2):
	"""Samples items from the store's validation speakers: strata {language: {group: count}}, at most per_speaker items
	per speaker, LibriVox / Audiobooks under 13 phonemes / s excluded as in training (overrunning segments). Prompts come
	from another utterance of the same speaker (not a near-duplicate line), cropped to prompt_s at a fixed offset."""
	from .data import StoreUtterances
	strata = strata or STRATA
	val = StoreUtterances(store, "val", groups=groups, min_s=min_s, max_s=max_s, min_rate=13, min_rate_groups=["LibriVox", "Audiobooks"])
	st = val.store
	rng = random.Random(seed)
	frames = st.frames[val.items]
	ok = (frames >= int(min_s * FPS)) & (frames <= int(max_s * FPS))
	items = []
	for code, groups_n in strata.items():
		lid = st.m["languages"].index(code)
		for gname, count in groups_n.items():
			gid = st.m["groups"].index(gname)
			pool = [int(u) for u in val.items[ok & (st.language[val.items] == lid) & (st.group[val.items] == gid)]]
			rng.shuffle(pool)
			taken, per = 0, {}
			for u in pool:
				if taken == count:
					break
				spk = int(st.speaker[u])
				if per.get(spk, 0) >= per_speaker:
					continue
				others = [int(v) for v in st.utterances_of(spk) if int(v) != u]
				key = _dup_key(st.text(u))
				others = [v for v in others if st.frames[v] >= int(prompt_s * FPS) and _dup_key(st.text(v)) != key]
				if not others:
					continue
				p = rng.choice(others)
				n = int(prompt_s * FPS)
				start = rng.randrange(int(st.frames[p]) - n + 1)
				items.append(dict(id=len(items), utt=u, language=code, group=gname, speaker=st.speaker_key(u), text=st.text(u),
					frames=int(st.frames[u]), prompt_utt=p, prompt_start=start, prompt_frames=n))
				taken += 1; per[spk] = per.get(spk, 0) + 1
			print(f"{code} {gname}: {taken} items")
	Path(out).parent.mkdir(parents=True, exist_ok=True)
	Path(out).write_text(json.dumps(dict(store=str(store), groups=groups, prompt_s=prompt_s, items=items), indent=1, ensure_ascii=False))
	print(f"wrote {len(items)} items to {out}")


def units(text, lang):
	from ..data import normalize_text
	from .sample import units as _units
	return _units(text, lang, normalize_text)


@torch.no_grad()
def run(setfile, ckpt, out, length="ref", length_scale=1.0, steps=25, cfg=2.5, seed=0, device="cuda", batch=16,
		prompt_s=None, temperature=1.0, gt="training/v3-evals/gt"):
	"""prompt_s: crop the prompts to this many seconds from their fixed offset instead of the set's (as much as the
	prompt utterance has). gt: the ground-truth run whose audio SIM is measured against."""
	from . import fsq
	from .model import V3, demask_flow, predict_length
	from .store import Store
	from ..emb.codecs.nemo import AudioCodecModel
	es = json.loads(Path(setfile).read_text())
	out = Path(out); out.mkdir(parents=True, exist_ok=True)
	st = Store(es["store"])
	if str(ckpt) == "gt":
		return run_gt(es, st, out, device, batch, setfile)
	sd = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
	model = V3(**sd["config"]).eval(); model.load_state_dict(sd["model"]); model.to(device)
	# the checkpoint's own phoneme ids, through the characters (store-trained checkpoints share the store's vocabulary)
	own = {c: i + 1 for i, c in enumerate(sd["vocab"])}
	vocab = st.m["vocab"]
	codec = AudioCodecModel.from_pretrained("nvidia/audio-codec-44khz").to(device).eval()
	digits_of = lambda codes: fsq.codes_to_digits(torch.from_numpy(np.asarray(codes).astype(np.int64)).t())
	(out / "settings.json").write_text(json.dumps(dict(set=str(setfile), ckpt=str(ckpt), length=length, length_scale=length_scale,
		steps=steps, cfg=cfg, seed=seed, prompt_s=prompt_s, temperature=temperature), indent=1))
	rows_path = out / "items.jsonl"
	done = {json.loads(l)["id"] for l in rows_path.read_text().splitlines()} if rows_path.exists() else set()
	import torchaudio
	with open(rows_path, "a") as rows:
		for it in es["items"]:
			if it["id"] in done:
				continue
			torch.manual_seed(seed * 100_003 + it["id"])
			text = torch.tensor([own[vocab[int(t)]] for t in st.phonemes(it["utt"]) if vocab[int(t)] in own], dtype=torch.long, device=device)
			pn = it["prompt_frames"] if prompt_s is None else int(prompt_s * FPS)
			pstart = min(it["prompt_start"], max(0, int(st.frames[it["prompt_utt"]]) - pn))
			pc = st.codes(it["prompt_utt"])[pstart : pstart + pn]
			prompt = to_patches(digits_of(pc), model.patch).to(device)
			lang = 0
			if getattr(model, "lang", None) is not None:
				lang = st.m["languages"].index(it["language"]) + 1
			with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
				if length == "predict":
					n = max(44, int(predict_length(model, text, prompt, lang=lang) * length_scale))
				else:
					n = int(it["frames"] * length_scale)
				gen = demask_flow(model, text, prompt, n, steps=steps, cfg=cfg, lang=lang, temperature=temperature)
			codes = fsq.digits_to_codes(gen).unsqueeze(0).to(device)
			wav = codec.decode(tokens=codes, tokens_len=torch.tensor([codes.shape[-1]], device=device))[0].float().cpu()
			torchaudio.save(str(out / f"{it['id']:04d}.wav"), wav, 44100)
			rows.write(json.dumps(dict(id=it["id"], language=it["language"], frames=n, ref_frames=it["frames"])) + "\n"); rows.flush()
			if it["id"] % 20 == 0:
				print(f"generated {it['id'] + 1}/{len(es['items'])}", flush=True)
	del model, codec
	torch.cuda.empty_cache()
	score(setfile, out, device, batch, gt=gt)


@torch.no_grad()
def run_gt(es, st, out, device, batch, setfile):
	"""The codec-decoded ground truth: its errors are the ceiling, and items whose reference is itself wrong (an
	overrunning segment, a bad transcript) can be excluded from comparisons with --gt."""
	import torchaudio
	from ..emb.codecs.nemo import AudioCodecModel
	codec = AudioCodecModel.from_pretrained("nvidia/audio-codec-44khz").to(device).eval()
	with open(out / "items.jsonl", "w") as rows:
		for it in es["items"]:
			codes = torch.from_numpy(np.asarray(st.codes(it["utt"])).astype(np.int64)).t()[None].to(device)
			wav = codec.decode(tokens=codes, tokens_len=torch.tensor([codes.shape[-1]], device=device))[0].float().cpu()
			torchaudio.save(str(out / f"{it['id']:04d}.wav"), wav, 44100)
			# the set's prompt clip, decoded: the fixed reference every run's SIM is measured against
			pc = torch.from_numpy(np.asarray(st.codes(it["prompt_utt"])[it["prompt_start"] : it["prompt_start"] + it["prompt_frames"]]).astype(np.int64)).t()[None].to(device)
			pw = codec.decode(tokens=pc, tokens_len=torch.tensor([pc.shape[-1]], device=device))[0].float().cpu()
			torchaudio.save(str(out / f"{it['id']:04d}_prompt.wav"), pw, 44100)
			rows.write(json.dumps(dict(id=it["id"], language=it["language"], frames=it["frames"], ref_frames=it["frames"])) + "\n")
	del codec
	torch.cuda.empty_cache()
	score(setfile, out, device, batch, gt=out)


def bad_references(gt, threshold=0.3):
	"""Item ids whose decoded ground truth itself scores worse than threshold."""
	if not gt:
		return set()
	return {s["id"] for s in map(json.loads, (Path(gt) / "scored.jsonl").read_text().splitlines()) if s["errors"] / max(1, s["units"]) > threshold}


def score(setfile, out, device="cuda", batch=16, gt="training/v3-evals/gt"):
	"""Transcribes every generated item in its language and writes scores.json. One file at a time, through the same
	helper as sample.py: batching short clips through the pipeline without an attention mask made Whisper loop
	(ground-truth French at 100% WER)."""
	from ..emb.transcribe import transcribe
	from .sample import _edits
	es = json.loads(Path(setfile).read_text())
	out = Path(out)
	gen = {json.loads(l)["id"]: json.loads(l) for l in (out / "items.jsonl").read_text().splitlines()}
	import torch.nn.functional as F
	from ..emb.similar import speaker_similarity_embedding
	emb = lambda p: speaker_similarity_embedding(p, device=device, dtype="float32")
	gt = Path(gt)
	# the references' embeddings are the same for every run: computed once, kept in the ground-truth run
	refs = {}
	if (gt / "ref_emb.pt").exists():
		refs = torch.load(gt / "ref_emb.pt")
	elif (gt / "0000_prompt.wav").exists():
		refs = {it["id"]: (emb(gt / f"{it['id']:04d}_prompt.wav").cpu(), emb(gt / f"{it['id']:04d}.wav").cpu()) for it in es["items"]}
		torch.save(refs, gt / "ref_emb.pt")
	scored = []
	for it in es["items"]:
		code = it["language"]
		g = emb(out / f"{it['id']:04d}.wav").cpu()
		# SIM against the set's prompt clip (SIM-O style; for the ground truth this is the ceiling) and against the
		# speaker's real recording of this sentence
		sim = F.cosine_similarity(g, refs[it["id"]][0], dim=-1).item() if it["id"] in refs else None
		sim_target = F.cosine_similarity(g, refs[it["id"]][1], dim=-1).item() if it["id"] in refs else None
		hyp = transcribe(out / f"{it['id']:04d}.wav", language=code, align=False, model_name="openai/whisper-large-v3-turbo",
			device=device, dtype="float16" if device == "cuda" else "float32")["text"]
		hu, ru = units(hyp, code), units(it["text"], code)
		scored.append(dict(id=it["id"], language=code, errors=_edits(hu, ru), units=len(ru), inserted=max(0, len(hu) - len(ru)),
			hyp=hyp.strip(), ref=it["text"], frames=gen[it["id"]]["frames"], ref_frames=it["frames"], sim=sim, sim_target=sim_target))
	with open(out / "scored.jsonl", "w") as f:
		for s in sorted(scored, key=lambda s: s["id"]):
			f.write(json.dumps(s, ensure_ascii=False) + "\n")
	res = summarize(scored)
	(out / "scores.json").write_text(json.dumps(res, indent=1))
	print_summary(res, out)


def _boot(rows, n=2000, seed=0):
	"""Pooled error rate and a 95% bootstrap interval over items."""
	e, u = np.array([r["errors"] for r in rows], float), np.array([r["units"] for r in rows], float)
	rng = np.random.default_rng(seed)
	idx = rng.integers(len(rows), size=(n, len(rows)))
	samples = e[idx].sum(1) / np.maximum(u[idx].sum(1), 1)
	return float(e.sum() / max(u.sum(), 1)), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def summarize(scored, exclude=()):
	"""Per language: the pooled error rate with a 95% bootstrap interval, the same with each item's errors capped at its
	length (one runaway loop can't dominate), inserted units, and how many items are over 50%."""
	scored = [s for s in scored if s["id"] not in exclude]
	res = {}
	langs = sorted({s["language"] for s in scored})
	for code in langs + ["all"]:
		rows = [s for s in scored if code == "all" or s["language"] == code]
		rate, lo, hi = _boot(rows)
		capped, clo, chi = _boot([dict(r, errors=min(r["errors"], r["units"])) for r in rows])
		res[code] = dict(rate=rate, ci=[lo, hi], capped=capped, capped_ci=[clo, chi], items=len(rows), metric="CER" if code in CER_LANGUAGES else ("mixed" if code == "all" else "WER"),
			inserted=sum(r["inserted"] for r in rows) / max(1, sum(r["units"] for r in rows)),
			over_50=sum(r["errors"] / max(1, r["units"]) > 0.5 for r in rows))
		for k in ("sim", "sim_target"):
			v = np.array([r[k] for r in rows if r.get(k) is not None])
			if len(v):
				boot = np.random.default_rng(0).choice(v, size=(2000, len(v))).mean(1)
				res[code][k] = [float(v.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
	return res


def print_summary(res, name=""):
	print(f"== {name}")
	for code, r in res.items():
		print(f"  {code:4s} {r['metric']:5s} {r['rate'] * 100:5.1f}%  [{r['ci'][0] * 100:5.1f}, {r['ci'][1] * 100:5.1f}]  "
			f"capped {r['capped'] * 100:5.1f}% [{r['capped_ci'][0] * 100:5.1f}, {r['capped_ci'][1] * 100:5.1f}]  "
			f"n={r['items']:3d}  inserted {r['inserted'] * 100:4.1f}%  items >50%: {r['over_50']}"
			+ (f"  SIM {r['sim'][0]:.3f} [{r['sim'][1]:.3f}, {r['sim'][2]:.3f}]" if "sim" in r else "")
			+ (f"  SIM-target {r['sim_target'][0]:.3f}" if "sim_target" in r else ""))


def compare(a, b, n=4000, seed=0, gt=None):
	"""Paired bootstrap of B - A per language (same items): the interval of the difference, and P(B better). gt: a
	ground-truth run, whose bad-reference items are left out."""
	load = lambda d: {s["id"]: s for s in map(json.loads, (Path(d) / "scored.jsonl").read_text().splitlines())}
	A, B = load(a), load(b)
	bad = bad_references(gt)
	ids = sorted((set(A) & set(B)) - bad)
	if bad:
		print(f"(leaving out {len(bad)} items whose ground truth scores over 30%)")
	print(f"== {b} vs {a} ({len(ids)} paired items; negative = B better)")
	rng = np.random.default_rng(seed)
	print("  (errors capped at 100% per item, so one runaway loop can't outweigh the rest)")
	for code in sorted({A[i]["language"] for i in ids}) + ["all"]:
		sel = [i for i in ids if code == "all" or A[i]["language"] == code]
		u = np.array([A[i]["units"] for i in sel], float)
		ea = np.minimum(np.array([A[i]["errors"] for i in sel], float), u)
		eb = np.minimum(np.array([B[i]["errors"] for i in sel], float), u)
		idx = rng.integers(len(sel), size=(n, len(sel)))
		d = (eb[idx].sum(1) - ea[idx].sum(1)) / np.maximum(u[idx].sum(1), 1)
		obs = (eb.sum() - ea.sum()) / max(u.sum(), 1)
		line = (f"  {code:4s} A {ea.sum() / u.sum() * 100:5.1f}%  B {eb.sum() / u.sum() * 100:5.1f}%  diff {obs * 100:+5.1f}  "
			f"[{np.percentile(d, 2.5) * 100:+5.1f}, {np.percentile(d, 97.5) * 100:+5.1f}]  P(B better) {np.mean(d < 0):.2f}")
		if all(A[i].get("sim") is not None and B[i].get("sim") is not None for i in sel):
			sa, sb = np.array([A[i]["sim"] for i in sel]), np.array([B[i]["sim"] for i in sel])
			ds = (sb - sa)[idx].mean(1)
			line += f"  | SIM A {sa.mean():.3f} B {sb.mean():.3f} diff {sb.mean() - sa.mean():+.3f} [{np.percentile(ds, 2.5):+.3f}, {np.percentile(ds, 97.5):+.3f}]"
		print(line)


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	sub = ap.add_subparsers(dest="cmd", required=True)
	b = sub.add_parser("build")
	b.add_argument("--store", type=Path, required=True)
	b.add_argument("--out", type=Path, required=True)
	b.add_argument("--groups", nargs="*", default=None)
	r = sub.add_parser("run")
	r.add_argument("--set", type=Path, required=True)
	r.add_argument("--ckpt", type=Path, required=True, help="a checkpoint, or `gt` for the decoded ground truth")
	r.add_argument("--out", type=Path, required=True)
	r.add_argument("--length", choices=["ref", "predict"], default="ref")
	r.add_argument("--length-scale", type=float, default=1.0)
	r.add_argument("--steps", type=int, default=25)
	r.add_argument("--cfg", type=float, default=2.5)
	r.add_argument("--seed", type=int, default=0)
	r.add_argument("--prompt-seconds", type=float, default=None, help="prompt length (default: the set's, 6 s)")
	r.add_argument("--temperature", type=float, default=1.0)
	s = sub.add_parser("score", help="re-score an existing run")
	s.add_argument("--set", type=Path, required=True)
	s.add_argument("--out", type=Path, required=True)
	t = sub.add_parser("report", help="summaries of runs, leaving out bad-reference items")
	t.add_argument("runs", type=Path, nargs="+")
	t.add_argument("--gt", type=Path, default=None)
	c = sub.add_parser("compare")
	c.add_argument("a", type=Path)
	c.add_argument("b", type=Path)
	c.add_argument("--gt", type=Path, default=None, help="a `run --ckpt gt` directory: leave out items with bad references")
	args = ap.parse_args()
	if args.cmd == "build":
		build(args.store, args.out, groups=args.groups)
	elif args.cmd == "run":
		run(args.set, args.ckpt, args.out, args.length, args.length_scale, args.steps, args.cfg, args.seed,
			prompt_s=args.prompt_seconds, temperature=args.temperature)
	elif args.cmd == "score":
		score(args.set, args.out)
	elif args.cmd == "report":
		bad = bad_references(args.gt)
		for d in args.runs:
			print_summary(summarize([json.loads(l) for l in (d / "scored.jsonl").read_text().splitlines()], bad), f"{d} (without {len(bad)} bad references)")
	else:
		compare(args.a, args.b, gt=args.gt)
