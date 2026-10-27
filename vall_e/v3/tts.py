# Inference with a v3 checkpoint: text (+ a voice prompt) -> 44.1 kHz audio.
#
#   python -m vall_e.v3.tts --ckpt training/v3-main5p-len/last.pt --text "Hello there." --prompt voice.wav --out out.wav
#
#   tts = TTS("training/v3-main5p-len/last.pt")
#   wav = tts.generate("Hello there.", prompt="voice.wav", language="en")   # [1, samples] at tts.sample_rate
#
# Text is phonemized as the dataset was (vall_e.emb.g2p), the prompt is encoded with the NeMo codec (cropped to the 10 s
# the model was fine-tuned on), the length comes from the model's length head unless a duration is given, and long
# texts are generated sentence by sentence (the model was trained on targets up to 12 s) and joined with short pauses.
import argparse
import re
from pathlib import Path

import torch
import torchaudio

from . import fsq
from .data import FPS, to_patches
from .model import V3, demask_flow, predict_length

PROMPT_S = 10.0  # the longest prompt in training
MAX_PHONEMES = 160  # about 10 s of speech: longer texts are split into sentences


class TTS:
	def __init__(self, ckpt, device="cuda"):
		st = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
		self.model = V3(**st["config"]).eval()
		self.model.load_state_dict(st["model"])
		self.model.to(device)
		self.vocab = {c: i + 1 for i, c in enumerate(st["vocab"])}  # 0 is padding
		self.device = device
		self.sample_rate = 44100
		from ..emb.codecs.nemo import AudioCodecModel
		self.codec = AudioCodecModel.from_pretrained("nvidia/audio-codec-44khz").to(device).eval()

	def phonemes(self, text, language="en"):
		from ..emb.g2p import encode
		ph = encode(text, language=language)
		ids = [self.vocab[c] for c in ph if c in self.vocab]
		return torch.tensor(ids, dtype=torch.long, device=self.device)

	@torch.no_grad()
	def encode_prompt(self, path, seconds=PROMPT_S):
		"""A voice prompt as model patches: mono, resampled to 44.1 kHz, the first `seconds`."""
		wav, sr = torchaudio.load(str(path))
		wav = wav.mean(0, keepdim=True)
		if sr != self.sample_rate:
			wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
		wav = wav[:, : int(seconds * self.sample_rate)].to(self.device)
		codes, n = self.codec.encode(audio=wav, audio_len=torch.tensor([wav.shape[-1]], device=self.device))
		digits = fsq.codes_to_digits(codes[0, :, : int(n[0])].long())
		return to_patches(digits, self.model.patch)

	@torch.no_grad()
	def decode(self, digits):
		codes = fsq.digits_to_codes(digits).unsqueeze(0).to(self.device)
		wav, _ = self.codec.decode(tokens=codes, tokens_len=torch.tensor([codes.shape[-1]], device=self.device))
		return wav.float().cpu()

	@torch.no_grad()
	def generate(self, text, prompt=None, language="en", duration=None, steps=25, cfg=2.5, flow_steps=8, sway=-1.0,
			temperature=1.0, seed=None, pause=0.25, length_scale=1.2, trim=True):
		"""prompt: a wav path (or None for an unconditioned voice). duration (s): the whole utterance's length, else
		predicted per sentence and multiplied by length_scale (a little under 1 if outputs trail off into extra words,
		as an overestimated length gets filled with speech). The default 1.2 suits models trained with silence padding
		(--pad-p), which fill spare time with silence; trim drops that trailing silence. Returns [1, samples]."""
		if seed is not None:
			torch.manual_seed(seed)
		p = self.encode_prompt(prompt) if prompt else torch.zeros(0, self.model.values, dtype=torch.long, device=self.device)
		parts = self._split(text, language)
		total_ph = sum(len(t) for t in parts)
		out = []
		for t in parts:
			with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
				if duration is not None:
					n = max(44, int(duration * FPS * len(t) / max(1, total_ph)))
				elif getattr(self.model, "len_q", None) is not None:
					n = max(44, int(predict_length(self.model, t, p) * length_scale))
				else:
					raise ValueError("this checkpoint has no length head: pass a duration")
				digits = demask_flow(self.model, t, p, n, steps=steps, cfg=cfg, temperature=temperature, flow_steps=flow_steps, sway=sway)
			w = self.decode(digits)
			out.append(_trim_silence(w) if trim else w)
		gap = torch.zeros(1, int(pause * self.sample_rate))
		return torch.cat([x for w in out for x in (w, gap)][:-1], dim=-1)

	def _split(self, text, language):
		"""Phoneme ids per sentence, merging sentences up to MAX_PHONEMES."""
		ids = self.phonemes(text, language)
		if len(ids) <= MAX_PHONEMES:
			return [ids]
		parts, cur = [], ""
		for s in re.split(r"(?<=[.!?。！？])\s*", text.strip()):
			if cur and len(self.phonemes(cur + " " + s, language)) > MAX_PHONEMES:
				parts.append(cur); cur = s
			else:
				cur = f"{cur} {s}".strip()
		if cur:
			parts.append(cur)
		return [self.phonemes(p, language) for p in parts if p.strip()]


def _trim_silence(wav, threshold=0.003, window=882, keep=0.05):
	"""Drops trailing 20 ms windows quieter than -50 dBFS, keeping `keep` seconds after the last loud one."""
	n = wav.shape[-1] // window * window
	rms = wav[..., :n].reshape(-1, window).pow(2).mean(-1).sqrt()
	loud = (rms >= threshold).nonzero()
	if len(loud) == 0:
		return wav
	end = min(wav.shape[-1], (int(loud[-1]) + 1) * window + int(keep * 44100))
	return wav[..., :end]


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--ckpt", type=Path, required=True)
	ap.add_argument("--text", required=True)
	ap.add_argument("--prompt", type=Path, default=None, help="voice prompt wav (up to 10 s is used); omit for no prompt")
	ap.add_argument("--language", default="en")
	ap.add_argument("--duration", type=float, default=None, help="seconds (default: predicted)")
	ap.add_argument("--out", type=Path, default=Path("out.wav"))
	ap.add_argument("--steps", type=int, default=25)
	ap.add_argument("--cfg", type=float, default=2.5)
	ap.add_argument("--seed", type=int, default=None)
	ap.add_argument("--length-scale", type=float, default=1.2, help="scale the predicted length (slack for silence-padded models)")
	ap.add_argument("--device", default="cuda")
	args = ap.parse_args()
	tts = TTS(args.ckpt, args.device)
	wav = tts.generate(args.text, args.prompt, args.language, args.duration, steps=args.steps, cfg=args.cfg, seed=args.seed,
		length_scale=args.length_scale)
	torchaudio.save(str(args.out), wav, tts.sample_rate)
	print(f"wrote {args.out} ({wav.shape[-1] / tts.sample_rate:.1f} s)")


if __name__ == "__main__":
	main()
