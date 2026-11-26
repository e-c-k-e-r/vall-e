# A fixed evaluation set for checkpoints, with confidence intervals. The earlier checks (6 items per language,
# 16 English) disagreed with each other by more than the differences being judged; this one freezes its inputs so runs
# are paired, and reports bootstrap intervals so a difference can be told from noise.
#
#   python -m vall_e.evalset build --store <root> --out evalsets/v3-eval.json            # once
#   python -m vall_e.evalset run --set evalsets/v3-eval.json --ckpt gt --out training/evals/gt                # once
#   python -m vall_e.evalset run --set evalsets/v3-eval.json --ckpt <ckpt> --out <dir> --length predict --cfg-prompt 2.0
#   python -m vall_e.evalset compare <dir A> <dir B>
#
# Items: validation speakers only (never trained on), English 100 + 30 per other language by default, each with a fixed
# prompt (another utterance of the same speaker, a fixed crop). Scoring: Whisper large-v3-turbo in the item's language,
# WER (CER for zh / ja / ko) as in sample.py, pooled over the units of each language.
import argparse
import json
import os
import random
from pathlib import Path

# the codec, Whisper and WavLM are all cached locally: don't let a network hiccup fail a run (one did, at
# huggingface.co, during v3-main6's step-200k eval)
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load_codec(device):
	"""NVIDIA's 44.1 kHz codec (vall_e/emb/codec.py; its cached copy when there is one, so it works offline)."""
	from .emb.codec import AudioCodecModel
	return AudioCodecModel.from_pretrained().to(device).eval()

# without it MIOpen (ROCm) benchmarks kernels for every new convolution shape (every utterance / clip length)
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

import numpy as np
import torch
from .audio import load as load_audio, save as save_audio

from .data import FPS, _dup_key, to_patches

CER_LANGUAGES = ("zh", "ja", "ko")


# English is stratified over its sources (drawn uniformly it was two-thirds Audiobooks, whose own transcripts are the
# least reliable); the other languages are all Emilia
STRATA = {"en": {"LibriTTS-R": 25, "LibriVox": 25, "Audiobooks": 25, "Emilia/EN": 25},
	"fr": {"Emilia/FR": 30}, "de": {"Emilia/DE": 30}, "ja": {"Emilia/JA": 30}, "zh": {"Emilia/ZH": 30}, "ko": {"Emilia/KO": 30}}


def build(store, out, strata=None, groups=None, prompt_s=6.0, seed=0, min_s=2.0, max_s=12.0, per_speaker=2, split="val", speakers=None):
	"""Samples items from the store's validation speakers: strata {language: {group: count}}, at most per_speaker items
	per speaker, LibriVox / Audiobooks under 13 phonemes / s excluded as in training (overrunning segments). Prompts come
	from another utterance of the same speaker (not a near-duplicate line), cropped to prompt_s at a fixed offset."""
	from .data import StoreUtterances
	strata = strata or STRATA
	val = StoreUtterances(store, split, groups=groups, min_s=min_s, max_s=max_s, min_rate=13, min_rate_groups=["LibriVox", "Audiobooks"])
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
				if per.get(spk, 0) >= per_speaker or ( speakers is not None and st.speaker_key( u ) not in speakers ):
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
	Path(out).write_text(json.dumps(dict(store=str(store), groups=groups, prompt_s=prompt_s, split=split, items=items), indent=1, ensure_ascii=False))
	print(f"wrote {len(items)} items to {out}")


def units(text, lang):
	from .emb.text import units as _units
	return _units(text, lang)


@torch.no_grad()
def run(setfile, ckpt, out, length="ref", length_scale=1.0, steps=25, cfg=6.0, seed=0, device="cuda", batch=16,
		prompt_s=None, temperature=1.0, gt="training/evals/gt", cfg_prompt=0.0, task="tts", snr=5.0):
	"""prompt_s: crop the prompts to this many seconds from their fixed offset instead of the set's (as much as the
	prompt utterance has). gt: the ground-truth run whose audio SIM is measured against. task: "tts"; "ns", noise
	suppression of the item's recording mixed with a held-out noise clip at `snr` dB (the same mixture every run), with
	its text; or "noisy", that mixture itself (the baseline ns should beat). ckpt is ignored for "noisy"."""
	from . import fsq
	from .model import V3, demask_flow, predict_length
	from .store import Store
	es = json.loads(Path(setfile).read_text())
	out = Path(out); out.mkdir(parents=True, exist_ok=True)
	st = Store(es["store"])
	if str(ckpt) == "gt":
		return run_gt(es, st, out, device, batch, setfile)
	if task == "noisy":
		return run_noisy(es, st, out, device, batch, setfile, snr, gt)
	from .export import load_checkpoint
	state, config, ck_vocab = load_checkpoint(ckpt)
	model = V3(**config).eval(); model.load_state_dict(state); model.to(device)
	# the checkpoint's own phoneme ids, through the characters (store-trained checkpoints share the store's vocabulary)
	own = {c: i + 1 for i, c in enumerate(ck_vocab)}
	vocab = st.m["vocab"]
	codec = load_codec(device)
	digits_of = lambda codes: fsq.codes_to_digits(torch.from_numpy(np.asarray(codes).astype(np.int64)).t())
	(out / "settings.json").write_text(json.dumps(dict(set=str(setfile), ckpt=str(ckpt), length=length, length_scale=length_scale,
		steps=steps, cfg=cfg, seed=seed, prompt_s=prompt_s, temperature=temperature, cfg_prompt=cfg_prompt, task=task, snr=snr), indent=1))
	noise = _eval_noise(es, codec, device) if task == "ns" else None
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
			if it.get("prompts"):
				# a multi-clip prompt: [utt, start, frames] clips joined with 0.15 s of the codec's digital silence
				sil = np.load(Path(es["store"]) / "silence.npy")[:1]
				parts = []
				for k, (pu, ps, pf) in enumerate(it["prompts"]):
					if k:
						parts.append(np.repeat(sil, int(0.15 * FPS), 0))
					parts.append(np.asarray(st.codes(pu)[ps : ps + pf]))
				pc = np.concatenate(parts)
			prompt = to_patches(digits_of(pc), model.patch).to(device)
			lang = 0
			if getattr(model, "lang", None) is not None:
				lang = st.m["languages"].index(it["language"]) + 1
			with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
				if task == "ns":
					from .model import NS
					mix = _mixture(st, codec, it, noise, snr, device)
					n = len(mix)
					gen = demask_flow(model, text, to_patches(mix, model.patch).to(device), n, steps=steps, cfg=cfg, lang=lang,
						temperature=temperature, mode=NS)
				else:
					if length == "predict":
						n = max(44, int(predict_length(model, text, prompt, lang=lang) * length_scale))
					else:
						n = int(it["frames"] * length_scale)
					gen = demask_flow(model, text, prompt, n, steps=steps, cfg=cfg, lang=lang, temperature=temperature, cfg_prompt=cfg_prompt)
			codes = fsq.digits_to_codes(gen).unsqueeze(0).to(device)
			wav = codec.decode(tokens=codes, tokens_len=torch.tensor([codes.shape[-1]], device=device))[0].float().cpu()
			save_audio(str(out / f"{it['id']:04d}.wav"), wav, 44100)
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
	codec = load_codec(device)
	with open(out / "items.jsonl", "w") as rows:
		for it in es["items"]:
			codes = torch.from_numpy(np.asarray(st.codes(it["utt"])).astype(np.int64)).t()[None].to(device)
			wav = codec.decode(tokens=codes, tokens_len=torch.tensor([codes.shape[-1]], device=device))[0].float().cpu()
			save_audio(str(out / f"{it['id']:04d}.wav"), wav, 44100)
			# the set's prompt clip, decoded: the fixed reference every run's SIM is measured against
			pc = torch.from_numpy(np.asarray(st.codes(it["prompt_utt"])[it["prompt_start"] : it["prompt_start"] + it["prompt_frames"]]).astype(np.int64)).t()[None].to(device)
			pw = codec.decode(tokens=pc, tokens_len=torch.tensor([pc.shape[-1]], device=device))[0].float().cpu()
			save_audio(str(out / f"{it['id']:04d}_prompt.wav"), pw, 44100)
			rows.write(json.dumps(dict(id=it["id"], language=it["language"], frames=it["frames"], ref_frames=it["frames"])) + "\n")
	del codec
	torch.cuda.empty_cache()
	score(setfile, out, device, batch, gt=out)


def _eval_noise(es, codec, device):
	"""The held-out noise clips (noisy.noise_split), decoded and normalised to unit power."""
	from .noisy import load_noise, noise_split, MAX_S
	clips = [c for i, c in enumerate(load_noise(es["store"])) if noise_split(i) == "eval"]
	out = []
	for c in clips:
		c = torch.from_numpy(np.asarray(c[: int(MAX_S * FPS)]).astype(np.int64)).t()[None].to(device)
		w = codec.decode(tokens=c, tokens_len=torch.tensor([c.shape[-1]], device=device))[0][0].float()
		if len(w) > 512:
			out.append(w / w.pow(2).mean().sqrt().clamp(min=1e-5))
	return out


@torch.no_grad()
def _mixture(st, codec, it, noise, snr, device):
	"""The item's recording mixed with a held-out noise clip at snr dB, encoded: [frames, 32] digits. Deterministic per
	item, so every run denoises the same input."""
	from . import fsq
	rng = np.random.default_rng(10_007 + it["id"])
	c = torch.from_numpy(np.asarray(st.codes(it["utt"])).astype(np.int64)).t()[None].to(device)
	clean = codec.decode(tokens=c, tokens_len=torch.tensor([c.shape[-1]], device=device))[0][0].float()
	nz = noise[int(rng.integers(len(noise)))]
	nz = nz.repeat(len(clean) // len(nz) + 2)
	o = int(rng.integers(len(nz) - len(clean)))
	mix = clean + nz[o : o + len(clean)] * (clean.pow(2).mean() / 10 ** (snr / 10)).sqrt()
	mix = (mix / mix.abs().max().clamp(min=1.0))[None]
	codes, n = codec.encode(audio=mix, audio_len=torch.tensor([mix.shape[-1]], device=device))
	return fsq.codes_to_digits(codes[0, :, : int(n[0])].long())


@torch.no_grad()
def run_noisy(es, st, out, device, batch, setfile, snr, gt):
	"""The noise-suppression baseline: the noisy mixtures themselves."""
	import torchaudio
	from . import fsq
	codec = load_codec(device)
	noise = _eval_noise(es, codec, device)
	with open(out / "items.jsonl", "w") as rows:
		for it in es["items"]:
			mix = _mixture(st, codec, it, noise, snr, device)
			codes = fsq.digits_to_codes(mix).unsqueeze(0).to(device)
			wav = codec.decode(tokens=codes, tokens_len=torch.tensor([codes.shape[-1]], device=device))[0].float().cpu()
			save_audio(str(out / f"{it['id']:04d}.wav"), wav, 44100)
			rows.write(json.dumps(dict(id=it["id"], language=it["language"], frames=len(mix), ref_frames=it["frames"])) + "\n")
	del codec
	torch.cuda.empty_cache()
	score(setfile, out, device, batch, gt=gt)


_UTMOS = {}


def utmos(path, device="cuda"):
	"""Predicted MOS (UTMOS22 strong, the VoiceMOS 2022 winner, via torch.hub tarepan/SpeechMOS): naturalness, 1-5."""
	import torchaudio
	if device not in _UTMOS:
		_UTMOS[device] = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()
	w, sr = load_audio(path)
	w = torchaudio.functional.resample(w.mean(0, keepdim=True), sr, 16000).to(device)
	with torch.no_grad():
		return float(_UTMOS[device](w, 16000).item())


def bandwidth(path, floor=50.0, split=12000.0):
	"""(cutoff in Hz: the highest frequency whose long-term level is within `floor` dB of the speech band's (1-4 kHz),
	power above `split` Hz relative to the total in dB). A 24 kHz source stops at 12 kHz and a lossy MP3 at its
	low-pass (11-16 kHz); most of the training audio is one or the other, so outputs may leave the codec's top octave
	empty."""
	import torchaudio
	w, sr = load_audio(path)
	w = w.mean(0)
	if len(w) < 2048:
		return 0.0, -99.0
	spec = torch.stft(w, 2048, 512, window=torch.hann_window(2048), return_complex=True).abs().pow(2).mean(-1)
	freqs = torch.linspace(0, sr / 2, len(spec))
	level = 10 * torch.log10(torch.nn.functional.avg_pool1d(spec[None, None], 9, 1, 4)[0, 0].clamp(min=1e-14))
	ref = level[(freqs >= 1000) & (freqs <= 4000)].median()
	above = (level > ref - floor).nonzero()
	cutoff = float(freqs[int(above.max())]) if len(above) else 0.0
	hf = float(10 * torch.log10(spec[freqs > split].sum().clamp(min=1e-12) / spec.sum().clamp(min=1e-12)))
	return cutoff, hf


def add_extras(out, device="cuda"):
	"""Adds UTMOS and bandwidth to a scored run (scored.jsonl), and refreshes its scores.json."""
	out = Path(out)
	rows = [json.loads(l) for l in (out / "scored.jsonl").read_text().splitlines()]
	for r in rows:
		f = out / f"{r['id']:04d}.wav"
		r["utmos"] = utmos(f, device)
		r["bw"], r["hf_db"] = bandwidth(f)
	with open(out / "scored.jsonl", "w") as fh:
		for r in rows:
			fh.write(json.dumps(r, ensure_ascii=False) + "\n")
	(out / "scores.json").write_text(json.dumps(summarize(rows), indent=1))
	return rows


def prosody_refs(setfile, gt, device="cuda"):
	"""Per item: the prompt clip's summary and articulation rate (units of its transcript per second of speech), and the
	ground truth's features and summary. Computed once, kept in the ground-truth run (prosody_ref.pt)."""
	from .emb import prosody as P
	gt = Path(gt)
	if (gt / "prosody_ref.pt").exists():
		return torch.load(gt / "prosody_ref.pt", weights_only=False)
	from .emb.asr import transcribe
	es = json.loads(Path(setfile).read_text())
	refs = {}
	for it in es["items"]:
		pf = gt / f"{it['id']:04d}_prompt.wav"
		ps = P.summary(P.features(pf, device))
		ps["rate"] = len(units(transcribe(pf, language=it["language"], device=device), it["language"])) / max(ps["speech"], 0.1)
		gf = P.features(gt / f"{it['id']:04d}.wav", device)
		refs[it["id"]] = dict(prompt=ps, gt=dict(feat=gf, summary=P.summary(gf)))
	torch.save(refs, gt / "prosody_ref.pt")
	return refs


def add_prosody(setfile, out, gt, device="cuda"):
	"""Adds prosody metrics (emb/prosody.py) to a scored run's rows ("pros") and refreshes its scores.json. The
	ground-truth run gets the prompt-based ones only (its own reading is the reference for the rest): they are the
	ceiling, the speaker's own variation between recordings."""
	from .emb import prosody as P
	out, gt = Path(out), Path(gt)
	refs = prosody_refs(setfile, gt, device)
	is_gt = out.resolve() == gt.resolve()
	rows = [json.loads(l) for l in (out / "scored.jsonl").read_text().splitlines()]
	for r in rows:
		ref = refs[r["id"]]
		r["pros"] = P.compare_to_refs(P.features(out / f"{r['id']:04d}.wav", device), r["units"], ref["prompt"], None if is_gt else ref["gt"])
	with open(out / "scored.jsonl", "w") as fh:
		for r in rows:
			fh.write(json.dumps(r, ensure_ascii=False) + "\n")
	(out / "scores.json").write_text(json.dumps(summarize(rows), indent=1))
	return rows


def bad_references(gt, threshold=0.3):
	"""Item ids whose decoded ground truth itself scores worse than threshold."""
	if not gt:
		return set()
	return {s["id"] for s in map(json.loads, (Path(gt) / "scored.jsonl").read_text().splitlines()) if s["errors"] / max(1, s["units"]) > threshold}


def score(setfile, out, device="cuda", batch=16, gt="training/evals/gt"):
	"""Transcribes every generated item in its language and writes scores.json. One file at a time, through the same
	helper as sample.py: batching short clips through the pipeline without an attention mask made Whisper loop
	(ground-truth French at 100% WER)."""
	from .emb.asr import transcribe
	from .emb.text import edits as _edits
	es = json.loads(Path(setfile).read_text())
	out = Path(out)
	gen = {json.loads(l)["id"]: json.loads(l) for l in (out / "items.jsonl").read_text().splitlines()}
	# transcription first, similarity after (the old vall_e.emb.similar, imported before the Whisper pipeline was built,
	# made every transcription take ~30 s instead of ~0.2 s; kept in this order to be safe)
	scored = []
	for it in es["items"]:
		code = it["language"]
		hyp = transcribe(out / f"{it['id']:04d}.wav", language=code, device=device)
		hu, ru = units(hyp, code), units(it["text"], code)
		scored.append(dict(id=it["id"], language=code, errors=_edits(hu, ru), units=len(ru), inserted=max(0, len(hu) - len(ru)),
			hyp=hyp.strip(), ref=it["text"], frames=gen[it["id"]]["frames"], ref_frames=it["frames"]))
	import torch.nn.functional as F
	from .emb.speaker import embedding
	emb = lambda p: embedding(p, device=device)
	gt = Path(gt)
	# the references' embeddings are the same for every run: computed once, kept in the ground-truth run
	refs = {}
	if (gt / "ref_emb.pt").exists():
		refs = torch.load(gt / "ref_emb.pt")
	elif (gt / "0000_prompt.wav").exists():
		refs = {it["id"]: (emb(gt / f"{it['id']:04d}_prompt.wav").cpu(), emb(gt / f"{it['id']:04d}.wav").cpu()) for it in es["items"]}
		torch.save(refs, gt / "ref_emb.pt")
	for row in scored:
		g = emb(out / f"{row['id']:04d}.wav").cpu()
		# SIM against the set's prompt clip (SIM-O style; for the ground truth this is the ceiling) and against the
		# speaker's real recording of this sentence
		row["sim"] = F.cosine_similarity(g, refs[row["id"]][0], dim=-1).item() if row["id"] in refs else None
		row["sim_target"] = F.cosine_similarity(g, refs[row["id"]][1], dim=-1).item() if row["id"] in refs else None
	for row in scored:
		f = out / f"{row['id']:04d}.wav"
		row["utmos"] = utmos(f, device)
		row["bw"], row["hf_db"] = bandwidth(f)
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
		bws = [r["bw"] for r in rows if r.get("bw") is not None]
		if bws:
			res[code]["bw_median"] = float(np.median(bws))
			res[code]["hf_db"] = float(np.mean([r["hf_db"] for r in rows if r.get("hf_db") is not None]))
		pros = {}
		for k in sorted({k for r in rows for k in r.get("pros", {})}):
			v = [r["pros"][k] for r in rows if r.get("pros", {}).get(k) is not None]
			if v:
				pros[k] = float(np.mean(v))
		if pros:
			res[code]["pros"] = pros
		for k in ("sim", "sim_target", "utmos"):
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
			+ (f"  SIM-target {r['sim_target'][0]:.3f}" if "sim_target" in r else "")
			+ (f"  UTMOS {r['utmos'][0]:.2f} [{r['utmos'][1]:.2f}, {r['utmos'][2]:.2f}]" if "utmos" in r else "")
			+ (f"  bandwidth {r['bw_median'] / 1000:.1f} kHz, >12 kHz {r['hf_db']:.0f} dB" if "bw_median" in r else ""))
		if "pros" in r:
			print("        prosody " + _pros_line(r["pros"]))


PROS_LABELS = dict(register_diff="register {:.2f} st", range_diff="range {:.2f} st", rate_vs_prompt="rate {:.3f}",
	dur_vs_gt="duration {:.3f}", pause_vs_gt="pauses {:.3f}", f0_corr="F0 r {:.3f}", f0_rmse="F0 {:.0f} c", energy_corr="energy r {:.3f}")


def _pros_line(p):
	"""register / range: |difference| from the prompt in semitones; rate: |log2 ratio| of articulation rate to the
	prompt's; duration: |log2 ratio| to the ground truth's; pauses: |difference| in pause share from the ground truth;
	F0 r / F0 c / energy r: aligned contour agreement with the ground truth (correlation, RMS difference in cents)."""
	return "  ".join(PROS_LABELS[k].format(p[k]) for k in PROS_LABELS if k in p)


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
		if all(A[i].get("utmos") is not None and B[i].get("utmos") is not None for i in sel):
			ua, ub = np.array([A[i]["utmos"] for i in sel]), np.array([B[i]["utmos"] for i in sel])
			du = (ub - ua)[idx].mean(1)
			line += f"  | UTMOS A {ua.mean():.2f} B {ub.mean():.2f} diff {ub.mean() - ua.mean():+.2f} [{np.percentile(du, 2.5):+.2f}, {np.percentile(du, 97.5):+.2f}]"
		print(line)
		from .emb.prosody import HIGHER
		parts = []
		for k in PROS_LABELS:
			ok = [i for i in sel if A[i].get("pros", {}).get(k) is not None and B[i].get("pros", {}).get(k) is not None]
			if len(ok) < 5:
				continue
			pa, pb = np.array([A[i]["pros"][k] for i in ok]), np.array([B[i]["pros"][k] for i in ok])
			dd = (pb - pa)[rng.integers(len(ok), size=(n, len(ok)))].mean(1)
			better = np.mean(dd > 0) if HIGHER[k] else np.mean(dd < 0)
			parts.append(f"{k} {pa.mean():.3f}->{pb.mean():.3f} P(B better) {better:.2f}")
		if parts:
			print("        prosody: " + "  ".join(parts))


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	sub = ap.add_subparsers(dest="cmd", required=True)
	b = sub.add_parser("build")
	b.add_argument("--store", type=Path, required=True)
	b.add_argument("--out", type=Path, required=True)
	b.add_argument("--groups", nargs="*", default=None)
	b.add_argument("--min-s", type=float, default=2.0)
	b.add_argument("--max-s", type=float, default=12.0)
	b.add_argument("--scale", type=float, default=1.0, help="scale every stratum's item count (e.g. 0.5 for a smaller long-form set)")
	b.add_argument("--seed", type=int, default=0)
	b.add_argument("--strata", default=None, help='JSON {language: {group: count}} instead of the default mix (e.g. a game-audio set)')
	b.add_argument("--per-speaker", type=int, default=2)
	b.add_argument("--speakers", type=Path, default=None, help='JSON list of "group/speaker" to draw from only (e.g. data/holdout-speakers.json)')
	b.add_argument("--prompt-s", type=float, default=6.0, help="prompt clip length (game lines are short: 6 s leaves several groups empty)")
	b.add_argument("--split", choices=["val", "train", "all"], default="val", help="speakers to draw from: validation only (unseen), or also / only trained ones")
	r = sub.add_parser("run")
	r.add_argument("--set", type=Path, required=True)
	r.add_argument("--ckpt", type=Path, required=True, help="a checkpoint, or `gt` for the decoded ground truth")
	r.add_argument("--out", type=Path, required=True)
	r.add_argument("--length", choices=["ref", "predict"], default="ref")
	r.add_argument("--length-scale", type=float, default=1.0)
	r.add_argument("--steps", type=int, default=25)
	r.add_argument("--cfg", type=float, default=6.0)
	r.add_argument("--seed", type=int, default=0)
	r.add_argument("--prompt-seconds", type=float, default=None, help="prompt length (default: the set's, 6 s)")
	r.add_argument("--temperature", type=float, default=1.0)
	r.add_argument("--cfg-prompt", type=float, default=0.0, help="extra speaker guidance (models trained with prompt-only dropout)")
	r.add_argument("--gt", type=Path, default=Path("training/evals/gt"), help="the ground-truth run SIM is measured against")
	r.add_argument("--task", choices=["tts", "ns", "noisy"], default="tts", help="ns: noise suppression of noised recordings; noisy: those recordings as they are")
	r.add_argument("--snr", type=float, default=5.0, help="for ns / noisy: the mixtures' SNR, dB")
	s = sub.add_parser("score", help="re-score an existing run")
	s.add_argument("--set", type=Path, required=True)
	s.add_argument("--out", type=Path, required=True)
	s.add_argument("--gt", type=Path, default=Path("training/evals/gt"))
	pr = sub.add_parser("prosody", help="add prosody metrics to scored runs (emb/prosody.py)")
	pr.add_argument("runs", nargs="+", type=Path)
	pr.add_argument("--set", type=Path, default=Path("evalsets/v3-eval.json"))
	pr.add_argument("--gt", type=Path, default=Path("training/evals/gt"))
	x = sub.add_parser("extras", help="add UTMOS and bandwidth to existing scored runs (no re-transcription)")
	x.add_argument("runs", type=Path, nargs="+")
	t = sub.add_parser("report", help="summaries of runs, leaving out bad-reference items")
	t.add_argument("runs", type=Path, nargs="+")
	t.add_argument("--gt", type=Path, default=None)
	c = sub.add_parser("compare")
	c.add_argument("a", type=Path)
	c.add_argument("b", type=Path)
	c.add_argument("--gt", type=Path, default=None, help="a `run --ckpt gt` directory: leave out items with bad references")
	args = ap.parse_args()
	if args.cmd == "build":
		base = json.loads(args.strata) if args.strata else STRATA
		strata = {l: {g: max(1, round(n * args.scale)) for g, n in gs.items()} for l, gs in base.items()}
		build(args.store, args.out, strata=strata, groups=args.groups, seed=args.seed, min_s=args.min_s, max_s=args.max_s, per_speaker=args.per_speaker, split=args.split, prompt_s=args.prompt_s,
			speakers=set(json.loads(args.speakers.read_text())) if args.speakers else None)
	elif args.cmd == "run":
		run(args.set, args.ckpt, args.out, args.length, args.length_scale, args.steps, args.cfg, args.seed,
			prompt_s=args.prompt_seconds, temperature=args.temperature, cfg_prompt=args.cfg_prompt, gt=args.gt, task=args.task, snr=args.snr)
	elif args.cmd == "score":
		score(args.set, args.out, gt=args.gt)
	elif args.cmd == "prosody":
		for d in args.runs:
			add_prosody(args.set, d, args.gt)
			print(f"{d}: prosody added")
	elif args.cmd == "extras":
		for d in args.runs:
			add_extras(d)
			print(f"{d}: UTMOS and bandwidth added")
	elif args.cmd == "report":
		bad = bad_references(args.gt)
		for d in args.runs:
			print_summary(summarize([json.loads(l) for l in (d / "scored.jsonl").read_text().splitlines()], bad), f"{d} (without {len(bad)} bad references)")
	else:
		compare(args.a, args.b, gt=args.gt)
