# A web page for inference: text + a voice prompt -> speech, on top of tts.py.
#
#   python -m vall_e.webui --ckpt training/v3-main6-len/last.pt [--models training] [--listen 0.0.0.0] [--port 7860] [--share]
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


def _checkpoints(root, current):
	"""Checkpoints under root (<run>/last.pt), most recent first, with the current one included."""
	found = sorted(Path(root).glob("*/last.pt"), key=lambda p: p.stat().st_mtime, reverse=True) if root else []
	names = [str(p) for p in found]
	return ([current] if current and current not in names else []) + names


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


def build(ckpt, models=None, device="cuda"):
	import gradio as gr
	with gr.Blocks(title="VALL-E") as app:
		gr.Markdown("# VALL-E\nText and a few seconds of a voice in, that voice speaking the text out (44.1 kHz).")
		with gr.Row():
			with gr.Column():
				text = gr.Textbox(label="Text", lines=4, placeholder="What should be said")
				prompt = gr.Audio(label="Voice prompt (about 6 s works best; up to 10 s is used)", sources=["upload", "microphone"], type="filepath")
				with gr.Row():
					language = gr.Dropdown(LANGUAGES, value="auto", label="Language")
					duration = gr.Number(value=0, label="Duration, s (0: predicted)", minimum=0)
				button = gr.Button("Generate", variant="primary")
			with gr.Column():
				audio = gr.Audio(label="Output", type="numpy", interactive=False)
				info = gr.Markdown()
				with gr.Accordion("Settings", open=False):
					checkpoint = gr.Dropdown(_checkpoints(models, ckpt), value=ckpt, label="Checkpoint", allow_custom_value=True)
					steps = gr.Slider(4, 64, value=25, step=1, label="Steps (WER is flat past ~25)")
					cfg = gr.Slider(1.0, 10.0, value=6.0, step=0.5, label="Guidance")
					cfg_prompt = gr.Slider(0.0, 4.0, value=2.0, step=0.25, label="Speaker guidance (0 for checkpoints trained without it)")
					length_scale = gr.Slider(0.7, 1.4, value=1.0, step=0.05, label="Length scale (speaking rate)")
					seed = gr.Number(value=-1, label="Seed (-1: random)", precision=0)
		button.click(lambda *a: synthesize(*a, device=device),
			[checkpoint, text, prompt, language, duration, steps, cfg, cfg_prompt, length_scale, seed], [audio, info])
	return app


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--ckpt", required=True, help="the checkpoint to start with")
	ap.add_argument("--models", default=None, help="a folder of runs (<run>/last.pt) to choose from in the page")
	ap.add_argument("--device", default="cuda")
	ap.add_argument("--listen", default="127.0.0.1")
	ap.add_argument("--port", type=int, default=7860)
	ap.add_argument("--share", action="store_true")
	args = ap.parse_args()
	_tts(args.ckpt, args.device)  # load before serving
	build(args.ckpt, args.models, args.device).queue().launch(server_name=args.listen, server_port=args.port, share=args.share)


if __name__ == "__main__":
	main()
