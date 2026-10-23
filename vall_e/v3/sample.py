# Generates validation utterances with a v3 checkpoint, decodes them with the NeMo codec, and optionally scores
# intelligibility (Whisper, in each item's language: WER, or CER for zh / ja / ko) and SIM-O against the prompt. The
# ground truth is decoded too, as the codec's ceiling.
import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

import torch
import torchaudio

from . import fsq
from .data import Utterances, to_patches
from .model import V3, demask, demask_flow, generate_ar, predict_length

CER_LANGUAGES = ("zh", "ja", "ko")  # scored per character: no (reliable) word boundaries


def _edits(a, b):
	prev = list(range(len(b) + 1))
	for i, x in enumerate(a, 1):
		cur = [i]
		for j, y in enumerate(b, 1):
			cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
		prev = cur
	return prev[-1]


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--ckpt", type=Path, required=True)
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--n", type=int, default=16)
	ap.add_argument("--steps", type=int, default=25)
	ap.add_argument("--cfg", type=float, default=2.5)
	ap.add_argument("--temperature", type=float, default=1.0)
	ap.add_argument("--score", action="store_true")
	# 8 sway-warped steps matched 16 uniform ones on v3-flow (WER 12.0% vs 12.4%, SIM-O 0.56 vs 0.53, 16 samples); 4 held up too
	ap.add_argument("--flow-steps", type=int, default=8)
	ap.add_argument("--sway", type=float, default=-1.0, help="flow time grid warp; -1 is F5-TTS's sway sampling, 0 uniform")
	ap.add_argument("--cfg-end", type=float, default=None, help="anneal CFG linearly to this over the demask steps")
	ap.add_argument("--temperature-end", type=float, default=None)
	ap.add_argument("--order", choices=["random", "vdc"], default="random")
	ap.add_argument("--mode", choices=["nar", "ar"], default="nar", help="ar: streaming-order generation (models trained with --ar-frac)")
	ap.add_argument("--cfg-mode", choices=["backbone", "head"], default="backbone", help="NAR CFG: drop text and prompt (two backbone passes) or the flow head's condition (one)")
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--prompt-seconds", type=float, default=None, help="fixed prompt length (default: 3-6 s, as in training)")
	ap.add_argument("--device", default="cuda", help="cpu works while a training run holds the GPU")
	ap.add_argument("--subset", type=int, default=0, help="sample the utterances of a `train.py --subset N` run")
	ap.add_argument("--null", choices=["both", "text", "prompt"], default="both",
		help="backbone CFG: what the unconditional pass drops (text / prompt alone need separate dropout in training)")
	ap.add_argument("--length", choices=["ref", "predict"], default="ref", help="NAR target length: the reference's, or LEN mode's prediction")
	ap.add_argument("--lang", default="auto", help="language id fed to accent models: auto (the item's), unknown, or a code (en, ja, ...)")
	ap.add_argument("--promptless", action="store_true", help="generate without a prompt (models trained with prompt dropout)")
	ap.add_argument("--per-language", type=int, default=0,
		help="with a store checkpoint: this many items per language instead of --n random ones (scored in their own language)")
	ap.add_argument("--swap-mode-embedding", action="store_true",
		help="exchange the AR and NAR-demask mode embeddings (attention masks unchanged), to test whether the model relies on them")
	args = ap.parse_args()
	torch.manual_seed(args.seed)
	device = args.device
	args.out.mkdir(parents=True, exist_ok=True)

	st = torch.load(args.ckpt, map_location=device, weights_only=False)
	model = V3(**st["config"]).to(device).eval()
	model.load_state_dict(st["model"])
	if args.swap_mode_embedding:
		from .model import AR, NAR_DEMASK
		with torch.no_grad():
			w = model.mode.weight
			w[[NAR_DEMASK, AR]] = w[[AR, NAR_DEMASK]].clone()
	from .data import StoreUtterances
	index = json.loads((args.ckpt.parent / "index.json").read_text()) if not st.get("store") else None
	prompt_kw = {"prompt_s": (args.prompt_seconds,) * 2} if args.prompt_seconds else {}
	languages = None
	if st.get("store"):
		val = StoreUtterances(st["store"], "val", seed=1, groups=st.get("groups"), languages=st.get("languages"), **prompt_kw)
		languages = val.store.m["languages"]
		if args.per_language:
			lang_of = val.store.language[val.items]
			picks = []
			for code in sorted(set(lang_of.tolist())):
				pool = [int(i) for i in (lang_of == code).nonzero()[0]]
				picks += random.Random(0).sample(pool, min(args.per_language, len(pool)))
			items = [val[i] for i in picks]
		else:
			items = [val[i] for i in random.Random(0).sample(range(len(val)), args.n)]
	elif args.subset:
		# the utterances `train.py --subset N` trained on, with the same prompts
		train = Utterances(index, "train")
		items = [train[i] for i in random.Random(0).sample(range(len(train)), args.subset)][: args.n]
	else:
		val = Utterances(index, "val", seed=1, **prompt_kw)
		items = [val[i] for i in random.Random(0).sample(range(len(val)), args.n)]

	from .. emb.codecs.nemo import AudioCodecModel
	codec = AudioCodecModel.from_pretrained("nvidia/audio-codec-44khz").to(device).eval()

	def decode(digits, path):
		codes = fsq.digits_to_codes(digits).unsqueeze(0).to(device)
		with torch.no_grad():
			wav, _ = codec.decode(tokens=codes, tokens_len=torch.tensor([codes.shape[-1]], device=device))
		torchaudio.save(str(path), wav.float().cpu(), 44100)

	def lang_id(it):
		if model.lang is None or args.lang == "unknown":
			return 0
		if args.lang == "auto":
			return it.get("lang", 0)
		return languages.index(args.lang) + 1

	rows = []
	for i, it in enumerate(items):
		prompt = to_patches(it["prompt"], model.patch).to(device)
		if args.promptless:
			prompt = prompt[:0]
		lang = lang_id(it)
		n_frames = len(it["target"])
		if args.length == "predict":
			with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
				n_frames = max(predict_length(model, it["text"].to(device), prompt, lang=lang), 44)  # at least 0.5 s (the codec's minimum is a few frames)
		with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
			if args.mode == "ar":
				# length from the stop logit; allow up to 1.5x the reference length
				gen = generate_ar(model, it["text"].to(device), prompt, int(n_frames * 1.5), cfg=args.cfg,
					temperature=args.temperature, flow_steps=args.flow_steps, sway=args.sway, min_frames=int(n_frames * 0.3), lang=lang)
			elif model.head_type == "flow":
				gen = demask_flow(model, it["text"].to(device), prompt, n_frames, steps=args.steps, cfg=args.cfg,
					temperature=args.temperature, flow_steps=args.flow_steps, sway=args.sway, cfg_end=args.cfg_end,
					temperature_end=args.temperature_end, order=args.order, cfg_mode=args.cfg_mode, lang=lang, null=args.null)
			else:
				gen = demask(model, it["text"].to(device), prompt, n_frames, steps=args.steps, cfg=args.cfg, temperature=args.temperature)
		decode(gen, args.out / f"{i:02d}_gen.wav")
		decode(it["target"], args.out / f"{i:02d}_ref.wav")
		decode(it["prompt"], args.out / f"{i:02d}_prompt.wav")  # the item's prompt, even when generating without it
		code = languages[it["lang"] - 1] if languages and it.get("lang") else "en"
		rows.append(dict(i=i, path=it["path"], speaker=it["speaker"], text=it.get("text_raw"), language=code,
			frames=len(gen), ref_frames=len(it["target"])))
	(args.out / "items.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
	print(f"wrote {len(rows)} samples to {args.out}")

	if args.score:
		score(args, rows, index)


def units(text, lang, normalize_text):
	"""Scoring units: words for English (the old normalisation: numbers and abbreviations expanded) and other spaced
	languages (lowercased, punctuation dropped), characters for zh / ja / ko."""
	if lang == "en":
		return normalize_text(text, language="en").split()
	text = "".join(" " if unicodedata.category(c)[0] in "PSZ" else c for c in unicodedata.normalize("NFKC", text).lower())
	if lang in CER_LANGUAGES:
		return list(re.sub(r"\s+", "", text))
	return text.split()


def score(args, rows, index):
	import torch.nn.functional as F
	from ..data import normalize_text
	from ..emb.transcribe import transcribe
	from ..emb.similar import speaker_similarity_embedding
	texts = {}
	for r in rows:
		if r.get("text"):
			texts[r["i"]] = r["text"]
			continue
		group, spk = r["speaker"].rsplit("/", 1)
		meta = Path("training/llama-nemo/metadata") / group / f"{spk}.json"
		texts[r["i"]] = json.loads(meta.read_text())[Path(r["path"]).stem]["text"]
	res = {}
	for kind in ("gen", "ref"):
		errs = {}  # language -> [edits, reference units]
		sims, sims_ref = [], []
		for r in rows:
			p = args.out / f"{r['i']:02d}_{kind}.wav"
			lang = r.get("language", "en")
			hyp = transcribe(p, language=lang, align=False, model_name="openai/whisper-large-v3-turbo", device=args.device, dtype="float16" if args.device == "cuda" else "float32")["text"]
			h, ref = units(hyp, lang, normalize_text), units(texts[r["i"]], lang, normalize_text)
			acc = errs.setdefault(lang, [0, 0])
			acc[0] += _edits(h, ref); acc[1] += len(ref)
			e = lambda x: speaker_similarity_embedding(x, device=args.device, dtype="float32")
			sims.append(F.cosine_similarity(e(p), e(args.out / f"{r['i']:02d}_prompt.wav"), dim=-1).item())
			# against the speaker's real recording of this sentence: unaffected by the prompt's length
			sims_ref.append(F.cosine_similarity(e(p), e(args.out / f"{r['i']:02d}_ref.wav"), dim=-1).item())
		word = [v for k, v in errs.items() if k not in CER_LANGUAGES]
		res[kind] = dict(sim=sum(sims) / len(sims), sim_ref=sum(sims_ref) / len(sims_ref))
		if word:  # over the word-scored languages (all of them for English-only runs, as before)
			res[kind] = dict(wer=sum(e for e, _ in word) / max(1, sum(n for _, n in word))) | res[kind]
		if len(errs) > 1 or set(errs) & set(CER_LANGUAGES):
			res[kind]["per_language"] = {k: round(e / max(1, n), 4) for k, (e, n) in sorted(errs.items())}
		print(kind, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[kind].items()}))
	(args.out / "scores.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
	main()
