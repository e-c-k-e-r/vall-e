# A web page for inference: text + a voice prompt -> speech, on top of tts.py.
#
#   python -m vall_e.webui [--ckpt <checkpoint or export>] [--models training] [--listen 0.0.0.0] [--port 7860] [--share]
import argparse
import os

# without it MIOpen (ROCm) benchmarks kernels for every new convolution shape, and the codec's shapes change with every
# utterance length: tts.py ran at 0.15-0.4x realtime instead of 1-4x
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
import random
import time
from pathlib import Path

import torch

from .tts import TTS

LANGUAGES = ["auto", "en", "fr", "de", "ja", "zh", "ko"]
_loaded = {}


def _tts(ckpt, device):
	"""One model loaded at a time (225M parameters, plus the codec)."""
	if ckpt not in _loaded:
		_loaded.clear()
		torch.cuda.empty_cache()
		_loaded[ckpt] = TTS(ckpt, device)
	return _loaded[ckpt]


def _is_export(path):
	"""A vall-e export (older .sft files, the codec's export and other formats aren't)."""
	from .export import FORMAT
	try:
		from safetensors import safe_open
		with safe_open(str(path), framework="pt") as f:
			return f.metadata().get("format") == FORMAT
	except Exception:
		return False


def _checkpoints(root, current):
	"""The released models (by name: downloaded on first use), the other vall-e exports in the models folder, then
	the runs under root (<run>/last.pt, most recent first); the current one included."""
	from .export import MODEL_DIR, RELEASES
	names = [r for r in RELEASES if r.startswith("nemo-44khz-nar-")]
	names += sorted(str(p) for p in MODEL_DIR.glob("*.sft") if p.name not in RELEASES and _is_export(p))
	if root:
		names += [str(p) for p in sorted(Path(root).glob("*/last.pt"), key=lambda p: p.stat().st_mtime, reverse=True)]
	return ([current] if current and current not in names else []) + names


def _harvard():
	"""A random Harvard sentence (IEEE 1969's phonetically balanced lists), the text box's starting text."""
	lines = [l.strip() for l in (Path(__file__).parent / "harvard_sentences.txt").read_text(encoding="utf-8").splitlines()]
	return random.choice([l for l in lines if l])


def synthesize(ckpt, text, prompt, language, duration, steps, cfg, cfg_prompt, length_scale, seed, device="cuda"):
	if not text or not text.strip():
		raise ValueError("enter some text")
	tts = _tts(ckpt, device)
	seed = int(seed) if seed is not None and seed >= 0 else random.randrange(2**31)
	parts = len(tts._split(text, language))
	t0 = time.time()
	wav = tts.generate(text, prompt or None, language, duration if duration and duration > 0 else None, steps=int(steps),
		cfg=cfg, cfg_prompt=cfg_prompt if prompt else 0.0, length_scale=length_scale, seed=seed)
	took = time.time() - t0
	seconds = wav.shape[-1] / tts.sample_rate
	info = (f"{seconds:.1f} s of audio in {took:.1f} s ({seconds / max(took, 1e-6):.2f}x realtime), "
		f"{parts} part{'s' if parts > 1 else ''}, seed {seed}" + ("" if prompt else "; no prompt: an unconditioned voice"))
	return (tts.sample_rate, wav[0].numpy()), info


def _guidance(ckpt):
	"""The guidance a checkpoint was tuned for (its export's metadata: v2.0 3.0 / 2.0, v2.0-fast 0 / 0), else the 225M's
	6.0 / 2.0, as tts.py defaults."""
	from .export import recommended_guidance
	g = recommended_guidance(ckpt) or dict(cfg=6.0, cfg_prompt=2.0)
	return g["cfg"], g["cfg_prompt"]


def build(ckpt, models=None, device="cuda"):
	import gradio as gr
	cfg0, cfg_prompt0 = _guidance(ckpt)
	with gr.Blocks(title="VALL-E") as app:
		gr.Markdown("# VALL-E\nText and a few seconds of a voice in, that voice speaking the text out (44.1 kHz).")
		with gr.Row():
			with gr.Column():
				text = gr.Textbox(label="Text", lines=4, value=_harvard, placeholder="What should be said")
				prompt = gr.Audio(label="Voice prompt (about 6 s works best; up to 10 s is used)", sources=["upload", "microphone"], type="filepath")
				with gr.Row():
					language = gr.Dropdown(LANGUAGES, value="en", label="Language (auto: detected, unreliable on short texts)")
					duration = gr.Number(value=0, label="Duration, s (0: predicted)", minimum=0)
				button = gr.Button("Generate", variant="primary")
			with gr.Column():
				audio = gr.Audio(label="Output", type="numpy", interactive=False)
				info = gr.Markdown()
				with gr.Accordion("Settings", open=False):
					checkpoint = gr.Dropdown(_checkpoints(models, ckpt), value=ckpt, label="Checkpoint", allow_custom_value=True)
					steps = gr.Slider(4, 64, value=25, step=1, label="Steps (WER is flat past ~25)")
					cfg = gr.Slider(0.0, 10.0, value=cfg0, step=0.5, label="Guidance (the checkpoint's own to start with)")
					cfg_prompt = gr.Slider(0.0, 4.0, value=cfg_prompt0, step=0.25, label="Speaker guidance (0 for checkpoints trained without it)")
					length_scale = gr.Slider(0.7, 1.4, value=1.0, step=0.05, label="Length scale (speaking rate)")
					seed = gr.Number(value=-1, label="Seed (-1: random)", precision=0)
		# a checkpoint chosen in the page brings its own guidance
		checkpoint.change(lambda c: tuple(gr.update(value=v) for v in _guidance(c)), [checkpoint], [cfg, cfg_prompt])
		button.click(lambda *a: synthesize(*a, device=device),
			[checkpoint, text, prompt, language, duration, steps, cfg, cfg_prompt, length_scale, seed], [audio, info])
	return app


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--ckpt", default=None, help="the checkpoint or export to start with (default: the default model)")
	ap.add_argument("--models", default=None, help="a folder of runs (<run>/last.pt) to choose from in the page")
	ap.add_argument("--device", default="cuda")
	ap.add_argument("--listen", default="127.0.0.1")
	ap.add_argument("--port", type=int, default=7860)
	ap.add_argument("--share", action="store_true")
	args = ap.parse_args()
	from .export import DEFAULT_MODEL
	ckpt = args.ckpt or DEFAULT_MODEL  # a released model's name: downloaded on first use
	_tts(ckpt, args.device)  # load before serving
	build(ckpt, args.models, args.device).queue().launch(server_name=args.listen, server_port=args.port, share=args.share)


if __name__ == "__main__":
	main()
