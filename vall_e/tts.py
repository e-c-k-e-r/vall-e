# Inference with a v3 checkpoint: text (+ a voice prompt) -> 44.1 kHz audio.
#
#   python -m vall_e.tts --text "Hello there." --prompt voice.wav --out out.wav     [--ckpt <checkpoint or export>]
#
#   tts = TTS()   # the default model; or TTS("<checkpoint or export>")
#   wav = tts.generate("Hello there.", prompt="voice.wav", language="en")   # [1, samples] at tts.sample_rate
#
# Text is phonemized as the dataset was (vall_e.emb.g2p), the prompt is encoded with the NeMo codec (cropped to the 10 s
# the model was fine-tuned on), the length comes from the model's length head unless a duration is given, and long
# texts are generated sentence by sentence (the model was trained on targets up to 12 s) and joined with short pauses.
import argparse
import os

# without it MIOpen (ROCm) benchmarks kernels for every new convolution shape, and the codec's shapes change with every
# utterance length: tts.py ran at 0.15-0.4x realtime instead of 1-4x
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
import re
from pathlib import Path

import torch
from .audio import load as load_audio, save as save_audio
import torchaudio

from . import fsq
from .data import FPS, to_patches
from .model import V3, demask_flow, predict_length

PROMPT_S = 10.0  # the longest prompt in training
MAX_PHONEMES = 160  # about 10 s of speech: longer texts are split into sentences


class TTS:
	def __init__(self, ckpt=None, device="cuda"):
		"""ckpt: a training checkpoint or an export (export.py); None for the default model (downloaded on first use)."""
		from .export import load_checkpoint, recommended_guidance
		from .evalset import load_codec
		state, config, vocab = load_checkpoint(ckpt)
		# the guidance the export was tuned for (v2.0: 3.0 / 2.0; a distilled model: 0 / 0), else the 225M models' 6.0 / 2.0
		g = recommended_guidance(ckpt) or dict(cfg=6.0, cfg_prompt=2.0)
		self.cfg, self.cfg_prompt = g["cfg"], g["cfg_prompt"]
		self.model = V3(**config).eval()
		self.model.load_state_dict(state)
		self.model.to(device)
		self.vocab = {c: i + 1 for i, c in enumerate(vocab)}  # 0 is padding
		self.device = device
		self.sample_rate = 44100
		self.codec = load_codec(device)

	def phonemes(self, text, language="en"):
		from .emb.g2p import encode
		ph = encode(text, language=language)
		ids = [self.vocab[c] for c in ph if c in self.vocab]
		return torch.tensor(ids, dtype=torch.long, device=self.device)

	def _load(self, path):
		wav, sr = load_audio(path)
		wav = wav.mean(0, keepdim=True)
		return torchaudio.functional.resample(wav, sr, self.sample_rate) if sr != self.sample_rate else wav

	@torch.no_grad()
	def voice_prompt(self, folder, clips=3, clip_s=3.2, pool=6, max_files=40):
		"""A multi-clip prompt from a folder of one voice's recordings: the `pool` clips closest to the voice's mean
		speaker embedding (the most representative), of those the `clips` with the best predicted MOS (the model copies
		the prompt's recording quality), each cut to clip_s and joined with 0.15 s of silence. On the eval set this beat
		a single random clip on similarity to the speaker's real recordings: +0.045 (+0.076 on held-out game voices)."""
		key = (str(Path(folder).resolve()), clips, clip_s, pool, max_files)
		if key in getattr(self, "_voices", {}):
			return self._voices[key]
		import random
		from .emb.speaker import embedding
		from .evalset import utmos
		files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".wav", ".mp3", ".ogg", ".flac", ".opus", ".m4a"))
		if not files:
			raise ValueError(f"no audio files in {folder}")
		if len(files) > max_files:
			files = random.Random(0).sample(files, max_files)
		embs = torch.stack([torch.nn.functional.normalize(embedding(p, device=self.device).float().reshape(-1), dim=0) for p in files])
		rep = (embs @ torch.nn.functional.normalize(embs.mean(0), dim=0)).cpu()
		best = rep.argsort(descending=True)[:pool].tolist()
		best = sorted(best, key=lambda i: -utmos(files[i], self.device))[:clips]
		gap = torch.zeros(1, int(0.15 * self.sample_rate))
		parts = []
		for k, i in enumerate(best):
			if k:
				parts.append(gap)
			parts.append(self._load(files[i])[:, : int(clip_s * self.sample_rate)])
		self._voices = getattr(self, "_voices", {})
		self._voices[key] = torch.cat(parts, -1)
		return self._voices[key]

	@torch.no_grad()
	def encode_prompt(self, path, seconds=PROMPT_S):
		"""A voice prompt as model patches: mono, resampled to 44.1 kHz, the first `seconds`. A folder makes a multi-clip
		prompt from several of its recordings (voice_prompt)."""
		wav = self.voice_prompt(path) if Path(path).is_dir() else self._load(path)
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
	def generate(self, text, prompt=None, language="en", duration=None, steps=25, cfg=None, flow_steps=8, sway=-1.0,
			temperature=1.0, seed=None, pause=0.25, length_scale=1.0, trim=True, cfg_prompt=None):
		"""prompt: a wav path (or None for an unconditioned voice). duration (s): the whole utterance's length, else
		predicted per sentence and multiplied by length_scale (a little under 1 if outputs trail off into extra words,
		as an overestimated length gets filled with speech). About 1.2 suits models trained with silence padding
		(--pad-p), which fill spare time with silence; trim drops that trailing silence. cfg_prompt: speaker guidance
		(models trained with prompt-only dropout, as v3-main6; eval set: SIM 0.700 -> 0.722 at 2.0, WER unchanged; set 0
		for checkpoints trained without it). Returns [1, samples]."""
		cfg = self.cfg if cfg is None else cfg
		cfg_prompt = self.cfg_prompt if cfg_prompt is None else cfg_prompt
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
				digits = demask_flow(self.model, t, p, n, steps=steps, cfg=cfg, temperature=temperature, flow_steps=flow_steps, sway=sway,
					cfg_prompt=cfg_prompt)
			w = self.decode(digits)
			out.append(_trim_silence(w) if trim else w)
		gap = torch.zeros(1, int(pause * self.sample_rate))
		return torch.cat([x for w in out for x in (w, gap)][:-1], dim=-1)

	@torch.no_grad()
	def encode_audio(self, path):
		"""A whole recording as frame digits [T, 32] (mono, 44.1 kHz)."""
		wav, sr = load_audio(path)
		wav = wav.mean(0, keepdim=True)
		if sr != self.sample_rate:
			wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
		wav = wav.to(self.device)
		codes, n = self.codec.encode(audio=wav, audio_len=torch.tensor([wav.shape[-1]], device=self.device))
		return fsq.codes_to_digits(codes[0, :, : int(n[0])].long())

	def _fill(self, text, language, known, keep, n, prompt, steps, cfg, cfg_prompt, seed, trim):
		if seed is not None:
			torch.manual_seed(seed)
		ids = self.phonemes(text, language)
		with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
			digits = demask_flow(self.model, ids, prompt, n, steps=steps, cfg=cfg, cfg_prompt=cfg_prompt, known=known, keep=keep)
		w = self.decode(digits)
		return _trim_silence(w) if trim else w

	@torch.no_grad()
	def continue_speech(self, audio, said, new, language="en", prompt=None, duration=None, steps=25, cfg=None, cfg_prompt=None,
			seed=None, trim=True):
		"""Continues a recording that says `said` with `new`: the recording is kept as is and the new part is generated in
		the same voice. duration (s): the new part's length, else predicted on its own (predicting the whole from the
		full text undershot and squeezed the new part). prompt: a voice clip, else the recording itself. Returns
		[1, samples], the recording included."""
		cfg = self.cfg if cfg is None else cfg
		cfg_prompt = self.cfg_prompt if cfg_prompt is None else cfg_prompt
		known = self.encode_audio(audio)
		p = self.encode_prompt(prompt) if prompt else to_patches(known[-int(PROMPT_S * FPS):], self.model.patch)
		with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
			extra = int(duration * FPS) if duration else predict_length(self.model, self.phonemes(new, language), p)
		n = len(known) + max(44, extra)
		full = torch.cat([known, known[-1:].expand(n - len(known), -1)])
		text = f"{said.strip()} {new.strip()}"
		return self._fill(text, language, full, torch.arange(n, device=full.device) < len(known), n, p, steps, cfg, cfg_prompt, seed, trim)

	@torch.no_grad()
	def edit(self, audio, text, start, end, old_text=None, language="en", prompt=None, steps=25, cfg=None, cfg_prompt=None,
			seed=None, trim=False):
		"""Re-speaks the part of a recording between `start` and `end` (seconds) so the whole says `text`. Without
		old_text (what the recording said) the span keeps its length; with it, the span is scaled by the change in
		phoneme count. prompt: a voice clip, else the recording itself. Returns [1, samples], the whole recording."""
		cfg = self.cfg if cfg is None else cfg
		cfg_prompt = self.cfg_prompt if cfg_prompt is None else cfg_prompt
		known = self.encode_audio(audio)
		T = len(known)
		s0, s1 = max(0, int(start * FPS)), min(T, int(end * FPS))
		span = s1 - s0
		if old_text:
			n_old, n_new = len(self.phonemes(old_text, language)), len(self.phonemes(text, language))
			span = max(4, span + int((n_new - n_old) * T / max(1, n_old)))
		p = self.encode_prompt(prompt) if prompt else to_patches(known[-int(PROMPT_S * FPS):], self.model.patch)
		full = torch.cat([known[:s0], known[:1].expand(span, -1), known[s1:]])
		keep = torch.ones(len(full), dtype=torch.bool, device=full.device)
		keep[s0 : s0 + span] = False
		return self._fill(text, language, full, keep, len(full), p, steps, cfg, cfg_prompt, seed, trim)

	@torch.no_grad()
	def stream(self, text, prompt=None, language="en", clauses=False, continuation=True, context_s=8.0, steps=25, cfg=None,
			cfg_prompt=None, length_scale=1.0, seed=None, pause=0.25):
		"""Yields the speech for `text` a chunk at a time ([1, samples] each), as soon as each is generated: chunks are
		sentences (clauses=True: also clauses, at , ; :). With continuation, each chunk is generated as the continuation
		of the chunks before it (their frames kept as known audio, up to context_s seconds of whole chunks, their text
		in front of the new chunk's), so voice and prosody carry across; without it, each chunk is generated on its own
		and chunks are joined with `pause` seconds of silence. Each chunk is decoded with up to a second of the audio
		before it, so the codec's decoder sees the join."""
		cfg = self.cfg if cfg is None else cfg
		cfg_prompt = self.cfg_prompt if cfg_prompt is None else cfg_prompt
		if seed is not None:
			torch.manual_seed(seed)
		p = self.encode_prompt(prompt) if prompt else None
		done = []  # (digits [T, 32], text) of the chunks so far
		chunks = _chunks(text, clauses)
		for k, chunk in enumerate(chunks):
			ids = self.phonemes(chunk, language)
			pp = p if p is not None else (to_patches(done[0][0], self.model.patch) if done else
				torch.zeros(0, self.model.values, dtype=torch.long, device=self.device))
			ctx, ctx_text = [], []
			if continuation:
				for d, t in reversed(done):
					if ctx and sum(len(x) for x in ctx) + len(d) > context_s * FPS:
						break
					ctx.insert(0, d); ctx_text.insert(0, t)
			with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
				n = max(44, int(predict_length(self.model, ids, pp) * length_scale))
				if ctx:
					known = torch.cat(ctx)
					full = torch.cat([known, known[-1:].expand(n, -1)])
					keep = torch.arange(len(full), device=full.device) < len(known)
					digits = demask_flow(self.model, self.phonemes(" ".join(ctx_text + [chunk]), language), pp, len(full), steps=steps,
						cfg=cfg, cfg_prompt=cfg_prompt, known=full, keep=keep)[len(known):]
				else:
					digits = demask_flow(self.model, ids, pp, n, steps=steps, cfg=cfg, cfg_prompt=cfg_prompt)
			tail = done[-1][0][-int(FPS):] if (done and continuation) else digits[:0]
			w = self.decode(torch.cat([tail, digits]))[:, len(tail) * 512 :]  # 512 samples per frame
			if k == len(chunks) - 1:
				w = _trim_silence(w)
			elif not continuation:
				w = torch.cat([_trim_silence(w), torch.zeros(1, int(pause * self.sample_rate))], -1)
			done.append((digits, chunk))
			yield w

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


def _chunks(text, clauses=False, min_words=3):
	"""Sentences (clauses=True: also clauses, split after , ; :), pieces under min_words merged into the next."""
	pat = r"(?<=[.!?。！？])\s+" if not clauses else r"(?<=[.!?。！？,;:、，；：])\s+"
	out, cur = [], ""
	for s in re.split(pat, text.strip()):
		cur = f"{cur} {s}".strip()
		if len(cur.split()) >= min_words:
			out.append(cur); cur = ""
	if cur:
		if out:
			out[-1] = f"{out[-1]} {cur}"
		else:
			out.append(cur)
	return out


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
	ap.add_argument("--ckpt", type=Path, default=None, help="a checkpoint or export (default: the default model, downloaded on first use)")
	ap.add_argument("--text", required=True)
	ap.add_argument("--prompt", type=Path, default=None, help="voice prompt: a wav (up to 10 s is used), or a folder of one voice's clips (the most representative, clean ones are joined); omit for no prompt")
	ap.add_argument("--language", default="en")
	ap.add_argument("--duration", type=float, default=None, help="seconds (default: predicted)")
	ap.add_argument("--out", type=Path, default=Path("out.wav"))
	ap.add_argument("--steps", type=int, default=25)
	ap.add_argument("--cfg", type=float, default=None, help="guidance strength (default: what the export was tuned for, else 6.0; 6.0 beat 2.5 on the 225M model: SIM +0.028, WER -2.6 points)")
	ap.add_argument("--seed", type=int, default=None)
	ap.add_argument("--cfg-prompt", type=float, default=None, help="speaker guidance (default: what the export was tuned for, else 2.0; 0 for checkpoints trained without prompt-only dropout)")
	ap.add_argument("--length-scale", type=float, default=1.0, help="scale the predicted length (about 1.2 for silence-padded models)")
	ap.add_argument("--device", default="cuda")
	args = ap.parse_args()
	tts = TTS(args.ckpt, args.device)
	wav = tts.generate(args.text, args.prompt, args.language, args.duration, steps=args.steps, cfg=args.cfg, seed=args.seed,
		length_scale=args.length_scale, cfg_prompt=args.cfg_prompt)
	save_audio(str(args.out), wav, tts.sample_rate)
	print(f"wrote {args.out} ({wav.shape[-1] / tts.sample_rate:.1f} s)")


if __name__ == "__main__":
	main()
