from ..config import cfg

import argparse
import random
import torch
import torchaudio

from functools import cache
from pathlib import Path


from encodec import EncodecModel
from encodec.utils import convert_audio
from einops import rearrange
from torch import Tensor
from tqdm import tqdm

try:
	from vocos import Vocos
except Exception as e:
	cfg.inference.use_vocos = False

@cache
def _load_encodec_model(device="cuda"):
	# Instantiate a pretrained EnCodec model
	assert cfg.sample_rate == 24_000

	# too lazy to un-if ladder this shit
	if cfg.models.prom_levels == 2:
		bandwidth_id = 1.5
	elif cfg.models.prom_levels == 4:
		bandwidth_id = 3.0
	elif cfg.models.prom_levels == 8:
		bandwidth_id = 6.0

	model = EncodecModel.encodec_model_24khz().to(device)
	model.set_target_bandwidth(bandwidth_id)
	model.bandwidth_id = bandwidth_id
	model.sample_rate = cfg.sample_rate
	model.backend = "encodec"

	return model

@cache
def _load_vocos_model(device="cuda"):
	assert cfg.sample_rate == 24_000

	model = Vocos.from_pretrained("charactr/vocos-encodec-24khz")
	model = model.to(device)

	# too lazy to un-if ladder this shit
	if cfg.models.prom_levels == 2:
		bandwidth_id = 0
	elif cfg.models.prom_levels == 4:
		bandwidth_id = 1
	elif cfg.models.prom_levels == 8:
		bandwidth_id = 2

	model.bandwidth_id = torch.tensor([bandwidth_id], device=device)
	model.sample_rate = cfg.sample_rate
	model.backend = "vocos"

	return model

@cache
def _load_model(device="cuda", vocos=cfg.inference.use_vocos):
	if vocos:
		model = _load_vocos_model(device)
	else:
		model = _load_encodec_model(device)

	return model

def unload_model():
	_load_model.cache_clear()
	_load_encodec_model.cache_clear()


@torch.inference_mode()
def decode(codes: Tensor, device="cuda"):
	"""
	Args:
		codes: (b q t)
	"""

	# expand if we're given a raw 1-RVQ stream
	if codes.dim() == 1:
		codes = rearrange(codes, "t -> 1 1 t")
	# expand to a batch size of one if not passed as a batch
	# vocos does not do batch decoding, but encodec does, but we don't end up using this anyways *I guess*
	# to-do, make this logical
	elif codes.dim() == 2:
		codes = rearrange(codes, "t q -> 1 q t")

	assert codes.dim() == 3, f'Requires shape (b q t) but got {codes.shape}'
	model = _load_model(device)

	# upcast so it won't whine
	if codes.dtype == torch.int8 or codes.dtype == torch.int16 or codes.dtype == torch.uint8:
		codes = codes.to(torch.int32)

	kwargs = {}
	if model.backend == "vocos":
		x = model.codes_to_features(codes[0])
		kwargs['bandwidth_id'] = model.bandwidth_id
	else:  
		x = [(codes.to(device), None)]

	wav = model.decode(x, **kwargs)

	if model.backend == "encodec":
		wav = wav[0]

	return wav, model.sample_rate

# huh
def decode_to_wave(resps: Tensor, device="cuda"):
	return decode(resps, device=device)

def decode_to_file(resps: Tensor, path: Path, device="cuda"):
	wavs, sr = decode(resps, device=device)

	torchaudio.save(str(path), wavs.cpu(), sr)
	return wavs, sr

def _replace_file_extension(path, suffix):
	return (path.parent / path.name.split(".")[0]).with_suffix(suffix)


@torch.inference_mode()
def encode(wav: Tensor, sr: int, device="cuda"):
	"""
	Args:
		wav: (t)
		sr: int
	"""

	model = _load_encodec_model(device)
	wav = wav.unsqueeze(0)
	wav = convert_audio(wav, sr, model.sample_rate, model.channels)
	wav = wav.to(device)

	encoded_frames = model.encode(wav)
	qnt = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # (b q t)

	return qnt


def encode_from_files(paths, device="cuda"):
	tuples = [ torchaudio.load(str(path)) for path in paths ]

	wavs = []
	main_sr = tuples[0][1]
	for wav, sr in tuples:
		assert sr == main_sr, "Mismatching sample rates"

		if wav.shape[0] == 2:
			wav = wav[:1]

		wavs.append(wav)

	wav = torch.cat(wavs, dim=-1)
	
	return encode(wav, sr, "cpu")

def encode_from_file(path, device="cuda"):
	if isinstance( path, list ):
		return encode_from_files( path, device )
	else:
		path = str(path)
		wav, sr = torchaudio.load(path, format=path[-3:])

	if wav.shape[0] == 2:
		wav = wav[:1]
	
	qnt = encode(wav, sr, device)

	return qnt


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("folder", type=Path)
	parser.add_argument("--suffix", default=".wav")
	args = parser.parse_args()

	paths = [*args.folder.rglob(f"*{args.suffix}")]

	for path in tqdm(paths):
		out_path = _replace_file_extension(path, ".qnt.pt")
		if out_path.exists():
			continue
		qnt = encode_from_file(path)
		torch.save(qnt.cpu(), out_path)


if __name__ == "__main__":
	main()
