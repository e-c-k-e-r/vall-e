from ..config import cfg

import argparse
import random
import math
import torch
import torchaudio
import numpy as np
import logging

_logger = logging.getLogger(__name__)

from functools import cache
from pathlib import Path
from typing import Union

from einops import rearrange
from torch import Tensor
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence

try:
	from .codecs.encodec import *
except Exception as e:
	cfg.inference.use_encodec = False
	_logger.warning(str(e))

try:
	from .codecs.vocos import *
except Exception as e:
	cfg.inference.use_vocos = False
	_logger.warning(str(e))

try:
	from .codecs.dac import *
except Exception as e:
	cfg.inference.use_dac = False
	_logger.warning(str(e))

try:
	from .codecs.nemo import *
except Exception as e:
	cfg.inference.use_nemo = False
	_logger.warning(str(e))

@cache
def _load_encodec_model(device="cuda", dtype=None, levels=0):
	assert cfg.sample_rate == 24_000

	if not levels:
		levels = cfg.model.max_levels

	# too lazy to un-if ladder this shit
	bandwidth_id = 6.0
	if levels == 2:
		bandwidth_id = 1.5
	elif levels == 4:
		bandwidth_id = 3.0
	elif levels == 8:
		bandwidth_id = 6.0

	# Instantiate a pretrained EnCodec model
	model = EncodecModel.encodec_model_24khz()
	model.set_target_bandwidth(bandwidth_id)
	
	model = model.to(device)
	model = model.eval()

	if dtype is not None:
		model = model.to(dtype)

	# extra metadata
	model.bandwidth_id = bandwidth_id
	model.normalize = cfg.inference.normalize
	model.backend = "encodec"

	return model

@cache
def _load_vocos_model(device="cuda", dtype=None, levels=0):
	assert cfg.sample_rate == 24_000

	if not levels:
		levels = cfg.model.max_levels

	model = Vocos.from_pretrained("charactr/vocos-encodec-24khz")
	model = model.to(device)
	model = model.eval()

	if dtype is not None:
		model = model.to(dtype)

	# too lazy to un-if ladder this shit
	bandwidth_id = 2
	if levels == 2:
		bandwidth_id = 0
	elif levels == 4:
		bandwidth_id = 1
	elif levels == 8:
		bandwidth_id = 2

	# extra metadata
	model.bandwidth_id = torch.tensor([bandwidth_id], device=device)
	model.backend = "vocos"

	return model

@cache
def _load_dac_model(device="cuda", dtype=None):
	kwargs = dict(model_type="44khz",model_bitrate="8kbps",tag="latest")
	# yes there's a better way, something like f'{cfg.sample.rate//1000}hz'		
	if cfg.sample_rate == 44_100:
		kwargs["model_type"] = "44khz"
	elif cfg.sample_rate == 16_000:
		kwargs["model_type"] = "16khz"
	else:
		raise Exception(f'unsupported sample rate: {cfg.sample_rate}')

	model = __load_dac_model(**kwargs)
	model = model.to(device)
	model = model.eval()

	if dtype is not None:
		model = model.to(dtype)

	model.backend = "dac"
	model.model_type = kwargs["model_type"]

	return model

@cache
def _load_nemo_model(device="cuda", dtype=None, model_name=None):
	if not model_name:
		model_name = "nvidia/audio-codec-44khz"

	model = AudioCodecModel.from_pretrained(model_name)
	model = model.to(device)
	model = model.eval()

	if dtype is not None:
		model = model.to(dtype)

	model.backend = "nemo"

	return model


@cache
def _load_model(device="cuda", backend=None, dtype=None):
	if not backend:
		backend = cfg.audio_backend

	if cfg.inference.amp:
		dtype = None

	if backend == "nemo":
		return _load_nemo_model(device, dtype=dtype)
	if backend == "audiodec":
		return _load_audiodec_model(device, dtype=dtype)
	if backend == "dac":
		return _load_dac_model(device, dtype=dtype)
	if backend == "vocos":
		return _load_vocos_model(device, dtype=dtype)

	return _load_encodec_model(device, dtype=dtype)

def unload_model():
	_load_model.cache_clear()
	_load_encodec_model.cache_clear() # because vocos can only decode

# to-do: clean up this mess
@torch.inference_mode()
def decode(codes: Tensor, device="cuda", dtype=None, metadata=None, window_duration=None):
	# upcast so it won't whine
	if codes.dtype in [torch.int8, torch.int16, torch.uint8]:
		codes = codes.to(torch.int32)

	# expand if we're given a raw 1-RVQ stream
	if codes.dim() == 1:
		codes = rearrange(codes, "t -> 1 1 t")

	# expand to a batch size of one if not passed as a batch
	elif codes.dim() == 2:
		# if (t, q), transpose to (q, t) instead
		if codes.shape[0] > codes.shape[1]:
			codes = codes.t()
		codes = codes.unsqueeze(0)

	# life is easier if we assume we're using a batch
	assert codes.dim() == 3, f'Requires shape (b q t) but got {codes.shape}'

	# load the model
	model = _load_model(device, dtype=dtype)
	# move to device
	codes = codes.to( device=device )

	# NeMo uses a different pathway
	if model.backend == "nemo":
		l = torch.tensor([c.shape[-1] for c in codes], device=device, dtype=torch.int32)
		wav, _ = model.decode(tokens=codes, tokens_len=l)
		return wav, cfg.sample_rate
	
	assert codes.shape[0] == 1, f'Batch decoding is unsupported for backend: {model.backend}'

	# DAC uses a different pathway
	if model.backend == "dac":
		dummy = False
		if metadata is None:
			metadata = dict(
				chunk_length=codes.shape[-1],
				original_length=0,
				input_db=-12,
				channels=1,
				sample_rate=cfg.sample_rate,
				padding=True,
				dac_version='1.0.0',
			)
			dummy = True
		elif hasattr( metadata, "__dict__" ):
			metadata = metadata.__dict__

		# generate object with copied metadata
		artifact = DACFile(
			codes = codes,
			chunk_length = math.floor(window_duration * cfg.dataset.frames_per_second) if window_duration else metadata["chunk_length"],
			original_length = metadata["original_length"],
			input_db = metadata["input_db"],
			channels = metadata["channels"],
			sample_rate = metadata["sample_rate"],
			padding = metadata["padding"],
			dac_version = metadata["dac_version"],
		)
		artifact.dummy = dummy

		# to-do: inject the sample rate encoded at, because we can actually decouple		
		return CodecMixin_decompress(model, artifact, verbose=False).audio_data[0], artifact.sample_rate

	# cleaner to separate out from EnCodec's pathway
	if model.backend == "vocos":
		x = model.codes_to_features(codes[0])
		wav = model.decode(x, bandwidth_id=model.bandwidth_id)
	
	if model.backend == "encodec":
		x = [(codes.to(device), None)]
		wav = model.decode(x)[0]

	return wav, cfg.sample_rate

@torch.inference_mode()
def decode_batch(codes: list[Tensor], device="cuda", dtype=None):
	# transpose if needed
	for i, code in enumerate(codes):
		if code.shape[0] < code.shape[1]:
			codes[i] = code.t()

	# store lengths
	lens = torch.tensor([code.shape[0] for code in codes], device=device, dtype=torch.int32)

	# pad and concat
	codes = pad_sequence(codes, batch_first=True)

	# re-transpose if needed
	if codes.shape[1] > codes.shape[2]:
		codes = rearrange(codes, "b t q -> b q t")

	# upcast so it won't whine
	if codes.dtype in [torch.int8, torch.int16, torch.uint8]:
		codes = codes.to(torch.int32)

	assert codes.dim() == 3, f'Requires shape (b q t) but got {codes.shape}'

	# load the model
	model = _load_model(device, dtype=dtype)
	# move to device
	codes = codes.to( device=device )

	# NeMo uses a different pathway
	if model.backend == "nemo":
		wav, lens = model.decode(tokens=codes, tokens_len=lens)
		return [ wav[:l].unsqueeze(0) for wav, l in zip(wav, lens) ], cfg.sample_rate

	# to-do: implement for encodec and vocos
	raise Exception(f"Batch decoding unsupported for backend {cfg.audio_backend}")

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
def encode(wav: Tensor, sr: int = cfg.sample_rate, device="cuda", dtype=None, return_metadata=True, window_duration=None):
	# expand if 1D
	if wav.dim() < 2:
		wav = wav.unsqueeze(0)
	# reshape (channels, samples) => (batch, channel, samples)
	if wav.dim() < 3:
		wav = wav.unsqueeze(0)

	if dtype is not None:
		wav = wav.to(dtype)

	# cringe assert
	assert wav.shape[0] == 1, f'Batch encoding is unsupported with vanilla encode()'
	
	model = _load_encodec_model( device, dtype=dtype ) if cfg.audio_backend == "vocos" else _load_model( device, dtype=dtype )

	# DAC uses a different pathway
	if cfg.audio_backend == "dac":
		signal = AudioSignal(wav, sample_rate=sr)
		
		artifact = model.compress(signal, win_duration=window_duration, verbose=False)
		return artifact.codes if not return_metadata else artifact
	
	# resample if necessary
	if sr != cfg.sample_rate or wav.shape[1] != 1:
		dtype = wav.dtype
		wav = convert_audio(wav.to(torch.float32), sr, cfg.sample_rate, 1).to(dtype)
	
	wav = wav.to(device)

	# NeMo uses a different pathway
	if cfg.audio_backend == "nemo":
		wav = wav.to(device)[:, 0, :]
		l = torch.tensor([w.shape[0] for w in wav]).to(device)
		with torch.autocast("cuda", dtype=cfg.inference.dtype, enabled=cfg.inference.amp):
			codes, lens = model.encode(audio=wav, audio_len=l)		
		# to-do: unpad 		
		return codes

	# vocos does not encode wavs to encodecs, so just use normal encodec
	if cfg.audio_backend in ["encodec", "vocos"]:
		with torch.autocast("cuda", dtype=cfg.inference.dtype, enabled=cfg.inference.amp):
			codes = model.encode(wav)
		codes = torch.cat([code[0] for code in codes], dim=-1)  # (b q t)
		return codes

@torch.inference_mode()
def encode_batch( wavs: list[Tensor], sr: list[int] | int = cfg.sample_rate, device="cuda", dtype=None ):
	# expand as list
	if not isinstance(sr, list):
		sr = [sr] * len(wavs)

	# resample if necessary
	for i, wav in enumerate(wavs):
		if sr[i] != cfg.sample_rate or wavs[i].shape[1] != 1:
			dtype = wav.dtype
			wavs[i] = convert_audio(wavs[i].to(torch.float32), sr[i], cfg.sample_rate, 1).to(dtype)

		# (frames) => (channel, frames)
		if wavs[i].dim() < 2:
			wavs[i] = wavs[i].unsqueeze(0)

		# transpose is required
		if wavs[i].shape[0] < wavs[i].shape[1]:
			wavs[i] = wavs[i].t()

	# store lengths
	lens = torch.tensor([wav.shape[0] for wav in wavs], device=device, dtype=torch.int32)

	# pad and concat (transpose because pad_sequence requires it this way)
	wav = pad_sequence(wavs, batch_first=True)
	# untranspose
	wav = rearrange(wav, "b t c -> b c t")
	#
	wav = wav.to(device)

	if dtype is not None:
		wav = wav.to(dtype)
	
	model = _load_encodec_model( device, dtype=dtype ) if cfg.audio_backend == "vocos" else _load_model( device, dtype=dtype )

	# NeMo uses a different pathway
	if cfg.audio_backend == "nemo":
		wav = wav.to(device)[:, 0, :]
		with torch.autocast("cuda", dtype=cfg.inference.dtype, enabled=cfg.inference.amp):
			codes, code_lens = model.encode(audio=wav, audio_len=lens)
		return [ code[:, :l] for code, l in zip( codes, code_lens ) ]

	# can't be assed to implement
	if cfg.audio_backend == "dac":
		raise Exception(f"Batch encoding unsupported for backend {cfg.audio_backend}")

	# naively encode
	if cfg.audio_backend in ["encodec", "vocos"]:
		with torch.autocast("cuda", dtype=cfg.inference.dtype, enabled=cfg.inference.amp):
			codes = model.encode(wav)
		codes = torch.cat([code[0] for code in codes], dim=-1)  # (b q t)

		return [ code[:, :l * cfg.dataset.frames_per_second // cfg.sample_rate] for code, l in zip(codes, lens) ]

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
	
	return encode(wav, sr, device)

def encode_from_file(path, device="cuda"):
	if isinstance( path, list ):
		return encode_from_files( path, device )
	else:
		path = str(path)
		wav, sr = torchaudio.load(path)

	if wav.shape[0] == 2:
		wav = wav[:1]
	
	qnt = encode(wav, sr, device)

	return qnt

"""
Helper Functions
"""

# DAC "silence": [ 568,  804,   10,  674,  364,  981,  568,  378,  731]

# trims from the start, up to `target`
def trim( qnt, target, reencode=False, device="cuda" ):
	length = max( qnt.shape[0], qnt.shape[1] )
	if target > 0:
		start = 0
		end = start + target
		if end >= length:
			start = length - target
			end = length
	# negative length specified, trim from end
	else:
		start = length + target
		end = length
		if start < 0:
			start = 0

	if not reencode:
		return qnt[start:end] if qnt.shape[0] > qnt.shape[1] else qnt[:, start:end]

	# trims on the waveform itself
	# need to test
	start = start / cfg.dataset.frames_per_second * cfg.sample_rate
	end = end / cfg.dataset.frames_per_second * cfg.sample_rate
	
	wav = decode(qnt, device=device)[0]
	return encode(wav[start:end], cfg.sample_rate, device=device)[0].t()

# trims a random piece of audio, up to `target`
# to-do: try and align to EnCodec window
def trim_random( qnt, target ):
	length = max( qnt.shape[0], qnt.shape[1] )
	start = int(length * random.random())
	end = start + target
	if end >= length:
		start = length - target
		end = length				

	return qnt[start:end] if qnt.shape[0] > qnt.shape[1] else qnt[:, start:end]

# repeats the audio to fit the target size
def repeat_extend_audio( qnt, target ):
	pieces = []
	length = 0
	while length < target:
		pieces.append(qnt)
		length += qnt.shape[0]

	return trim(torch.cat(pieces), target)

# interleaves between a list of audios
# useful for interleaving silence
def interleave_audio( *args, audio=None ):
	qnts = [ *args ]
	qnts = [ qnt for qnt in qnts if qnt is not None ]

	if audio is None:
		return qnts

	# interleave silence
	# yes there's a better way
	res = []
	for i, qnt in enumerate(qnts):
		res.append( qnt )
		if i + 1 != len(qnts):
			res.append( audio )

	return res

# concats two audios together
def concat_audio( *args, reencode=False, device="cuda" ):
	qnts = [ *args ]
	qnts = [ qnt for qnt in qnts if qnt is not None ]
	# just naively combine the codes
	if not reencode:
		return torch.concat( qnts )

	decoded = [ decode(qnt, device=device)[0] for qnt in qnts ]
	combined = torch.concat( decoded )
	return encode(combined, cfg.sample_rate, device=device)[0].t()

# merges two quantized audios together
# requires re-encoding because there's no good way to combine the waveforms of two audios without relying on some embedding magic
def merge_audio( *args, device="cuda", scale=[] ):
	qnts = [ *args ]
	qnts = [ qnt for qnt in qnts if qnt is not None ]
	decoded = [ decode(qnt, device=device)[0] for qnt in qnts ]

	# max length
	max_length = max([ wav.shape[-1] for wav in decoded ])
	for i, wav in enumerate(decoded):
		delta = max_length - wav.shape[-1]
		if delta <= 0:
			continue
		pad = torch.zeros( (1, delta), dtype=wav.dtype, device=wav.device )
		decoded[i] = torch.cat( [ wav, pad ], dim=-1 )

	# useful to adjust the volumes of each waveform
	if len(scale) == len(decoded):
		for i in range(len(scale)):
			decoded[i] = decoded[i] * scale[i]

	combined = sum(decoded) / len(decoded)
	return encode(combined, cfg.sample_rate, device=device)[0].t()

# Get framerate for a given audio backend
def get_framerate( backend=None, sample_rate=None ):
	if not backend:
		backend = cfg.audio_backend
	if not sample_rate:
		sample_rate = cfg.sample_rate

	if backend == "dac":
		if sample_rate == 44_100:
			return 87
		if sample_rate == 16_000:
			return 50
	
	# 24Khz Encodec / Vocos and incidentally DAC are all at 75Hz
	return 75

# Generates quantized silence
def get_silence( length, device=None, codes=None ):
	length = math.floor(length * get_framerate())
	if cfg.audio_backend == "dac":
		codes = [ 568, 804, 10, 674, 364, 981, 568, 378, 731 ]
	else:
		codes = [ 62, 424, 786, 673, 622, 986, 570, 948 ]

	return torch.tensor([ codes for _ in range( length ) ], device=device, dtype=torch.int16)

# Pads a sequence of codes with silence
def pad_codes_with_silence( codes, size=1 ):
	duration = codes.shape[0] * get_framerate()
	difference = math.ceil( duration + size ) - duration

	silence = get_silence( difference, device=codes.device )[:, :codes.shape[-1]]

	half = math.floor(difference / 2 * get_framerate())

	return torch.concat( [ silence[half:, :], codes, silence[:half, :] ], dim=0 )

# Generates an empty waveform
def get_silent_waveform( length, device=None ):
	length = math.floor(length * cfg.sample_rate)
	return torch.tensor( [ [ 0 for _ in range( length ) ] ], device=device, dtype=torch.float32 )

# Pads a waveform with silence
def pad_waveform_with_silence( waveform, sample_rate, size=1 ):
	duration = waveform.shape[-1] / sample_rate
	difference = math.ceil( duration + size ) - duration

	silence = get_silent_waveform( difference, device=waveform.device )

	half = math.floor(difference / 2 * sample_rate)

	return torch.concat( [ silence[:, half:], waveform, silence[:, :half] ], dim=-1 )

# Encodes/decodes audio, and helps me debug things
if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--audio-backend", type=str, default="encodec")
	parser.add_argument("--input", type=Path)
	parser.add_argument("--output", type=Path, default=None)
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--dtype", type=str, default="float16")
	parser.add_argument("--window-duration", type=float, default=None) # for DAC, the window duration for encoding / decoding
	parser.add_argument("--print", action="store_true") # prints codes and metadata
	parser.add_argument("--pad", action="store_true") # to test if padding with silence modifies the waveform / quants too much

	args = parser.parse_args()

	# prepare from args
	cfg.set_audio_backend(args.audio_backend)
	audio_extension = cfg.audio_backend_extension

	cfg.inference.weight_dtype = args.dtype # "bfloat16"
	cfg.inference.amp = args.dtype != "float32"
	cfg.device = args.device

	# decode
	if args.input.suffix == audio_extension:
		args.output = args.input.with_suffix('.wav') if not args.output else args.output.with_suffix('.wav')

		artifact = np.load(args.input, allow_pickle=True)[()]
		codes = torch.from_numpy(artifact['codes'])[0][:, :].t().to(device=cfg.device, dtype=torch.int16)

		# pad to nearest
		if args.pad:
			codes = pad_codes_with_silence( codes )
			del artifact['metadata']

		waveform, sample_rate = decode( codes, device=cfg.device, metadata=artifact['metadata'] if 'metadata' in artifact else None, window_duration=args.window_duration )

		torchaudio.save(args.output, waveform.cpu(), sample_rate)
		
		# print
		if args.print:
			torch.set_printoptions(profile="full")

			_logger.info(f"Metadata: {artifact['metadata']}" )
			_logger.info(f"Codes: {codes.shape}, {codes}" )
	# encode
	else:
		args.output = args.input.with_suffix(audio_extension) if not args.output else args.output.with_suffix(audio_extension)
		
		waveform, sample_rate = torchaudio.load(args.input)

		# pad to nearest
		if args.pad:
			waveform = pad_waveform_with_silence( waveform, sample_rate )
		
		qnt = encode(waveform.to(cfg.device), sr=sample_rate, device=cfg.device, window_duration=args.window_duration)

		if cfg.audio_backend == "dac":
			state_dict = {
				"codes": qnt.codes.cpu().numpy().astype(np.uint16),
				"metadata": {
					"original_length": qnt.original_length,
					"sample_rate": qnt.sample_rate,
					
					"input_db": qnt.input_db.cpu().numpy().astype(np.float32),
					"chunk_length": qnt.chunk_length,
					"channels": qnt.channels,
					"padding": qnt.padding,
					"dac_version": "1.0.0",
				},
			}
		else:						
			state_dict = {
				"codes": qnt.cpu().numpy().astype(np.uint16),
				"metadata": {
					"original_length": waveform.shape[-1],
					"sample_rate": sample_rate,
				},
			}
		np.save(open(args.output, "wb"), state_dict)