# Whisper transcription for scoring generated speech (WER / CER).
from functools import cache

import torch


@cache
def _pipeline(model_name="openai/whisper-large-v3-turbo", device="cuda", dtype="float16"):
	from transformers import pipeline
	return pipeline("automatic-speech-recognition", model=model_name, torch_dtype=getattr(torch, dtype), device=device,
		model_kwargs={"attn_implementation": "sdpa"})


def transcribe(path, language="en", model_name="openai/whisper-large-v3-turbo", device="cuda", dtype=None):
	"""Text of an audio file in `language` (None or "auto": detected). One file per call: batching short clips through
	the pipeline without an attention mask makes Whisper loop."""
	dtype = dtype or ("float16" if str(device).startswith("cuda") else "float32")
	asr = _pipeline(model_name, device, dtype)
	kwargs = {"task": "transcribe", "language": None if language in (None, "auto") else language}
	return asr(str(path), chunk_length_s=30, batch_size=16, generate_kwargs=kwargs)["text"].strip()
