# Audio files through soundfile (libsndfile, bundled with its wheels: WAV, FLAC, OGG / Opus, MP3), as torchaudio.load /
# torchaudio.save returned and wrote them: torchaudio 2.9 and later only read and write through torchcodec, which needs
# FFmpeg's libraries on the system. Formats libsndfile can't read (m4a / AAC) still go to torchaudio when it can.
import numpy as np
import torch


def load(path):
	"""-> ([channels, samples] float32 in [-1, 1], sample rate), as torchaudio.load."""
	import os
	import soundfile as sf
	if not os.path.exists(path):
		raise FileNotFoundError(path)
	try:
		data, sr = sf.read(str(path), dtype="float32", always_2d=True)
	except sf.LibsndfileError:
		import torchaudio
		return torchaudio.load(str(path))
	return torch.from_numpy(np.ascontiguousarray(data.T)), sr


def save(path, wav, sample_rate):
	"""wav: [channels, samples] or [samples] in [-1, 1] -> a file in the extension's format (16-bit for WAV / FLAC, as
	torchaudio.save wrote them)."""
	import soundfile as sf
	wav = torch.as_tensor(wav).detach().float().cpu()
	if wav.dim() == 1:
		wav = wav[None]
	pcm16 = str(path).lower().endswith((".wav", ".flac"))
	sf.write(str(path), wav.clamp(-1, 1).numpy().T, int(sample_rate), subtype="PCM_16" if pcm16 else None)
