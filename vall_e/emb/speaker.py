# Speaker similarity (SIM-O): cosine similarity of WavLM-large + ECAPA-TDNN speaker embeddings, the speaker
# verification model common to TTS evaluations (UniAudio's fine-tuned weights).
import os
from functools import cache
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

WEIGHTS_URL = "https://huggingface.co/Dongchao/UniAudio/resolve/main/wavlm_large_finetune.pth"
MODEL_DIR = Path(os.environ.get("VALL_E_MODELS", Path(__file__).resolve().parents[2] / "data" / "models"))


def _download(path, url=WEIGHTS_URL):
	import requests
	path.parent.mkdir(parents=True, exist_ok=True)
	with requests.get(url, stream=True) as r:
		r.raise_for_status()
		with open(path.with_suffix(".tmp"), "wb") as f:
			for chunk in r.iter_content(1 << 20):
				f.write(chunk)
	path.with_suffix(".tmp").replace(path)


@cache
def _model(device="cuda", dtype="float32"):
	from .ecapa_tdnn import ECAPA_TDNN_SMALL
	model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large")
	weights = MODEL_DIR / "wavlm_large_finetune.pth"
	if not weights.exists():
		_download(weights)
	state = torch.load(weights, map_location="cpu")["model"]
	del state["loss_calculator.projection.weight"]
	model.load_state_dict(state)
	return model.to(device=device, dtype=getattr(torch, dtype)).eval()


@torch.no_grad()
def embedding(path, device="cuda", dtype="float32"):
	"""Speaker embedding of an audio file (mono, resampled to 16 kHz)."""
	wav, sr = torchaudio.load(str(path))
	wav = torchaudio.transforms.Resample(sr, 16000)(wav.mean(0, keepdim=True))
	return _model(device, dtype)(wav.to(device=device, dtype=getattr(torch, dtype)))


def similarity(a, b, device="cuda"):
	"""Cosine similarity of two files' (or embeddings') speaker embeddings."""
	ea = a if torch.is_tensor(a) else embedding(a, device)
	eb = b if torch.is_tensor(b) else embedding(b, device)
	return F.cosine_similarity(ea.to(eb.device), eb, dim=-1).item()
