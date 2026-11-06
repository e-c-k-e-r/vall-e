# Golden references for ports of the inference path (volta-engine): for a few fixed texts and prompts, everything an
# engine has to reproduce, with the randomness supplied as data so it doesn't need torch's generator.
#
#   python -m vall_e.golden --ckpt data/models/nemo-44khz-nar-flow-225m-v1.2.sft --out training/golden/v1.2
#
# Per case, <out>/<case>.safetensors holds:
#   text [Lt] phoneme ids, prompt [Lp, 64] prompt patches (digits), lang, n_frames (from the length head, fp32) and
#   log_patches (its raw output);
#   order [N] the reveal order, noise_<s> [k_s, 64] the flow head's starting noise at demask step s;
#   step0_hidden_<l> [3, L, d] the input to block l (0..layers) at demask step 0, for the cond / null / promptless rows
#   (l = layers is the final norm's output; fp16), step0_v_<j> [k_0, 64] the guided velocity at flow step j of demask step 0;
#   digits_<s> [La, 64] the patches after demask step s; digits [n_frames, 32] the result; codes [8, n_frames];
#   wav [samples] the decoded audio (44.1 kHz).
# A JSON sidecar has the settings and the case texts. The loop here mirrors model.demask_flow (backbone CFG with
# speaker guidance, or none); main() checks that it reproduces demask_flow's output for the same seed.
import argparse
import json
import math
from pathlib import Path

import torch

from . import fsq
from .data import pack
from .model import NAR_DEMASK, flow_sample, predict_length, snap

CASES = [
	dict(name="en-short", text="The quick brown fox jumps over the lazy dog.", language="en"),
	dict(name="en-long", text="It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.", language="en"),
	dict(name="de", text="Das Wetter ist heute schön, also gehen wir am Nachmittag spazieren.", language="de"),
]


@torch.no_grad()
def reference(model, text, prompt, n_frames, order, noises, steps=25, cfg=6.0, cfg_prompt=2.0, flow_steps=8, sway=-1.0, lang=0, dump=None):
	"""demask_flow's default path with the reveal order and the per-step flow noise given. dump: a dict to fill with the
	step-0 block inputs and guided velocities and every step's patches."""
	dev = text.device
	La = math.ceil(n_frames / model.patch)
	digits = torch.zeros(La, model.values, dtype=torch.long, device=dev)
	masked = torch.ones(La, dtype=torch.bool, device=dev)
	guided = cfg > 1
	rows = 3 if guided and cfg_prompt > 0 else 2 if guided else 1
	texts = [text, text[:0], text][:rows]
	prompts = [prompt, prompt[:0], prompt[:0]][:rows]
	done = 0
	for s in range(steps):
		target = La - int(math.floor(math.cos((s + 1) / steps * math.pi / 2) * La))
		reveal = order[done:target]
		if len(reveal) == 0:
			continue
		b = pack(texts, prompts, [digits] * rows, [masked] * rows, langs=[lang] * rows)
		if dump is not None and s == 0:
			hooks = [blk.register_forward_pre_hook(lambda m, a, l=l: dump.__setitem__(f"step0_hidden_{l}", a[0].half().clone()))
				for l, blk in enumerate(model.blocks)]
		h_all = model.hidden(b, NAR_DEMASK)
		if dump is not None and s == 0:
			for hk in hooks:
				hk.remove()
			dump[f"step0_hidden_{len(model.blocks)}"] = h_all.view(rows, -1, h_all.shape[-1]).half().clone()
		h = h_all[b["tgt_index"]].view(rows, La, -1)[:, reveal]
		x0 = noises[s].to(dev)
		vals = _flow(model, h, x0, guided, cfg, cfg_prompt, flow_steps, sway, dump if s == 0 else None)
		digits[reveal] = snap(vals, model.patch)
		masked[reveal] = False
		if dump is not None:
			dump[f"digits_{s}"] = digits.clone()
		done = target
	return digits.view(-1, fsq.SCALARS)[:n_frames]


def _flow(model, h, x, guided, cfg, cfg_prompt, steps, sway, dump):
	"""flow_sample's Euler loop from the given starting noise x."""
	grid = torch.linspace(0, 1, steps + 1).tolist()
	if sway:
		grid = [u + sway * (math.cos(math.pi / 2 * u) - 1 + u) for u in grid]
	n = h.shape[1]
	for k in range(steps):
		t = torch.full((n,), grid[k], device=x.device)
		if guided:
			r = h.shape[0]
			v = model.flow(torch.cat([x] * r).to(h.dtype), torch.cat([t] * r), h.flatten(0, 1)).float().view(r, n, -1)
			v = v[1] + (v[0] - v[1]) * cfg + ((v[0] - v[2]) * cfg_prompt if r == 3 else 0)
		else:
			v = model.flow(x.to(h.dtype), t, h[0]).float()
		if dump is not None:
			dump[f"step0_v_{k}"] = v.clone()
		x = x + v * (grid[k + 1] - grid[k])
	return x


def draws(model, n_frames, steps, seed, device):
	"""The reveal order and per-step noise in the order demask_flow draws them (randperm, then one randn per step)."""
	torch.manual_seed(seed)
	La = math.ceil(n_frames / model.patch)
	order = torch.randperm(La, device=device)
	noises, done = [], 0
	for s in range(steps):
		target = La - int(math.floor(math.cos((s + 1) / steps * math.pi / 2) * La))
		k = target - done
		noises.append(torch.randn(k, model.values, device=device) if k else torch.zeros(0, model.values, device=device))
		done = target
	return order, noises


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--ckpt", default=None)
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--prompt", default="training/evals/gt/0003_prompt.wav")
	ap.add_argument("--cfg", type=float, default=6.0)
	ap.add_argument("--cfg-prompt", type=float, default=2.0)
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--fp32", action="store_true", help="compute in FP32 (the export's FP16 weights, upcast) instead of bf16 autocast: closer to an engine's FP16 GEMMs with FP32 accumulation")
	args = ap.parse_args()
	from safetensors.torch import save_file
	from .model import demask_flow
	from .tts import TTS
	tts = TTS(args.ckpt)
	model, dev = tts.model, tts.device
	args.out.mkdir(parents=True, exist_ok=True)
	prompt = tts.encode_prompt(args.prompt)
	meta = dict(ckpt=str(args.ckpt), prompt=args.prompt, cfg=args.cfg, cfg_prompt=args.cfg_prompt, steps=25, flow_steps=8, sway=-1.0,
		seed=args.seed, precision="fp32" if args.fp32 else "bf16 autocast", config=model.config, cases=CASES)
	for case in CASES:
		text = tts.phonemes(case["text"], case["language"])
		amp = lambda: torch.autocast(dev, dtype=torch.bfloat16, enabled=not args.fp32)
		with amp():
			n = predict_length(model, text, prompt)
		order, noises = draws(model, n, 25, args.seed, dev)
		dump = {}
		with amp():
			digits = reference(model, text, prompt, n, order, noises, cfg=args.cfg, cfg_prompt=args.cfg_prompt, dump=dump)
			torch.manual_seed(args.seed)
			check = demask_flow(model, text, prompt, n, cfg=args.cfg, cfg_prompt=args.cfg_prompt)
		same = float((check == digits).float().mean())
		wav = tts.decode(digits)[0]
		t = dict(text=text, prompt=prompt, n_frames=torch.tensor(n), order=order, digits=digits, codes=fsq.digits_to_codes(digits), wav=wav,
			**{f"noise_{s}": x for s, x in enumerate(noises)}, **dump)
		save_file({k: v.detach().cpu().contiguous() for k, v in t.items()}, str(args.out / f"{case['name']}.safetensors"))
		print(f"{case['name']}: {n} frames; matches demask_flow on {same:.1%} of digits", flush=True)
		case["n_frames"], case["matches_demask_flow"] = n, same
	(args.out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))


if __name__ == "__main__":
	main()
