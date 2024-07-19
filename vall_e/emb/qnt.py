from ..config import cfg

import argparse
import random
import torch
import torchaudio

from functools import cache
from pathlib import Path
from typing import Union

from einops import rearrange
from torch import Tensor
from tqdm import tqdm

try:
	from encodec import EncodecModel
	from encodec.utils import convert_audio
except Exception as e:
	cfg.inference.use_encodec = False

try:
	from vocos import Vocos
except Exception as e:
	cfg.inference.use_vocos = False

try:
	from dac import DACFile
	from audiotools import AudioSignal
	from dac.utils import load_model as __load_dac_model

	"""
	Patch decode to skip things related to the metadata (namely the waveform trimming)
	So far it seems the raw waveform can just be returned without any post-processing
	A smart implementation would just reuse the values from the input prompt
	"""
	from dac.model.base import CodecMixin

	@torch.no_grad()
	def CodecMixin_decompress(
		self,
		obj: Union[str, Path, DACFile],
		verbose: bool = False,
	) -> AudioSignal:
		self.eval()
		if isinstance(obj, (str, Path)):
			obj = DACFile.load(obj)

		original_padding = self.padding
		self.padding = obj.padding

		range_fn = range if not verbose else tqdm.trange
		codes = obj.codes
		original_device = codes.device
		chunk_length = obj.chunk_length
		recons = []

		for i in range_fn(0, codes.shape[-1], chunk_length):
			c = codes[..., i : i + chunk_length].to(self.device)
			z = self.quantizer.from_codes(c)[0]
			r = self.decode(z)
			recons.append(r.to(original_device))

		recons = torch.cat(recons, dim=-1)
		recons = AudioSignal(recons, self.sample_rate)

		# to-do, original implementation
		if not hasattr(obj, "dummy") or not obj.dummy:
			resample_fn = recons.resample
			loudness_fn = recons.loudness
			
			# If audio is > 10 minutes long, use the ffmpeg versions
			if recons.signal_duration >= 10 * 60 * 60:
				resample_fn = recons.ffmpeg_resample
				loudness_fn = recons.ffmpeg_loudness

			recons.normalize(obj.input_db)
			resample_fn(obj.sample_rate)
			recons = recons[..., : obj.original_length]
			loudness_fn()
			recons.audio_data = recons.audio_data.reshape(
				-1, obj.channels, obj.original_length
			)
		self.padding = original_padding
		return recons

	CodecMixin.decompress = CodecMixin_decompress

except Exception as e:
	cfg.inference.use_dac = False
	print(str(e))

# uses https://github.com/facebookresearch/AudioDec/
# I have set up a pip-ify'd version with the caveat of having to manually handle downloading the checkpoints with a wget + unzip
# I was not happy with testing, it sounded rather mediocre.
"""
try:
	from audiodec.utils.audiodec import AudioDec, assign_model as _audiodec_assign_model
except Exception as e:
	cfg.inference.use_audiodec = False
	print(str(e))
"""

@cache
def _load_encodec_model(device="cuda", levels=cfg.model.max_levels):
	assert cfg.sample_rate == 24_000

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

	# extra metadata
	model.bandwidth_id = bandwidth_id
	model.sample_rate = cfg.sample_rate
	model.normalize = cfg.inference.normalize
	model.backend = "encodec"

	return model

@cache
def _load_vocos_model(device="cuda", levels=cfg.model.max_levels):
	assert cfg.sample_rate == 24_000

	model = Vocos.from_pretrained("charactr/vocos-encodec-24khz")
	model = model.to(device)
	model = model.eval()

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
	model.sample_rate = cfg.sample_rate
	model.backend = "vocos"

	return model

@cache
def _load_dac_model(device="cuda", levels=cfg.model.max_levels):
	kwargs = dict(model_type="44khz",model_bitrate="8kbps",tag="latest")
	if not cfg.variable_sample_rate:
		# yes there's a better way, something like f'{cfg.sample.rate//1000}hz'
		if cfg.sample_rate == 44_000 or cfg.sample_rate == 44_100: # because I messed up and had assumed it was an even 44K and not 44.1K
			kwargs["model_type"] = "44khz"
		elif cfg.sample_rate == 16_000:
			kwargs["model_type"] = "16khz"
		else:
			raise Exception(f'unsupported sample rate: {cfg.sample_rate}')

	model = __load_dac_model(**kwargs)
	model = model.to(device)
	model = model.eval()

	# to revisit later, but experiments shown that this is a bad idea
	if cfg.variable_sample_rate:
		model.sample_rate = cfg.sample_rate

	model.backend = "dac"
	model.model_type = kwargs["model_type"]

	return model

@cache
def _load_audiodec_model(device="cuda", model_name=None, levels=cfg.model.max_levels):
	if not model_name:
		model_name = "libritts_v1" if cfg.sample_rate == 24_000 else "vctk_v1"
	sample_rate, encoder_checkpoint, decoder_checkpoint = _audiodec_assign_model(model_name)

	model = AudioDec(tx_device=device , rx_device=device )
	model.load_transmitter(encoder_checkpoint)
	model.load_receiver(encoder_checkpoint, decoder_checkpoint)

	model.backend = "audiodec"
	model.sample_rate = sample_rate

	return model

@cache
def _load_model(device="cuda", backend=None, levels=cfg.model.max_levels):
	if not backend:
		backend = cfg.audio_backend

	if backend == "audiodec":
		return _load_audiodec_model(device, levels=levels)
	if backend == "dac":
		return _load_dac_model(device, levels=levels)
	if backend == "vocos":
		return _load_vocos_model(device, levels=levels)

	return _load_encodec_model(device, levels=levels)

def unload_model():
	_load_model.cache_clear()
	_load_encodec_model.cache_clear() # because vocos can only decode

@torch.inference_mode()
def decode(codes: Tensor, device="cuda", levels=cfg.model.max_levels, metadata=None):
	# upcast so it won't whine
	if codes.dtype == torch.int8 or codes.dtype == torch.int16 or codes.dtype == torch.uint8:
		codes = codes.to(torch.int32)

	# expand if we're given a raw 1-RVQ stream
	if codes.dim() == 1:
		codes = rearrange(codes, "t -> 1 1 t")
	# expand to a batch size of one if not passed as a batch
	# vocos does not do batch decoding, but encodec does, but we don't end up using this anyways *I guess*
	# to-do, make this logical
	elif codes.dim() == 2:
		codes = rearrange(codes, "t q -> 1 q t")

	assert codes.dim() == 3, f'Requires shape (b q t) but got {codes.shape}'

	# load the model
	model = _load_model(device, levels=levels)

	# AudioDec uses a different pathway
	if model.backend == "audiodec":
		codes = codes.to( device=device )[0]
		zq = model.rx_encoder.lookup( codes )
		wav = model.decoder.decode(zq).squeeze(1)
		return wav, model.sample_rate

	# DAC uses a different pathway
	if model.backend == "dac":
		dummy = False
		if metadata is None:
			metadata = dict(
				chunk_length= codes.shape[-1],
				original_length=0,
				input_db=-12,
				channels=1,
				sample_rate=model.sample_rate,
				padding=True,
				dac_version='1.0.0',
			)
			dummy = True
		# generate object with copied metadata
		artifact = DACFile(
			codes = codes,
			# yes I can **kwargs from a dict but what if I want to pass the actual DACFile.metadata from elsewhere
			chunk_length = metadata["chunk_length"] if isinstance(metadata, dict) else metadata.chunk_length,
			original_length = metadata["original_length"] if isinstance(metadata, dict) else metadata.original_length,
			input_db = metadata["input_db"] if isinstance(metadata, dict) else metadata.input_db,
			channels = metadata["channels"] if isinstance(metadata, dict) else metadata.channels,
			sample_rate = metadata["sample_rate"] if isinstance(metadata, dict) else metadata.sample_rate,
			padding = metadata["padding"] if isinstance(metadata, dict) else metadata.padding,
			dac_version = metadata["dac_version"] if isinstance(metadata, dict) else metadata.dac_version,
		)
		artifact.dummy = dummy

		# to-do: inject the sample rate encoded at, because we can actually decouple		
		return CodecMixin_decompress(model, artifact, verbose=False).audio_data[0], artifact.sample_rate

	kwargs = {}
	if model.backend == "vocos":
		x = model.codes_to_features(codes[0])
		kwargs['bandwidth_id'] = model.bandwidth_id
	else:  
		# encodec will decode as a batch
		x = [(codes.to(device), None)]

	wav = model.decode(x, **kwargs)

	# encodec will decode as a batch
	if model.backend == "encodec":
		wav = wav[0]

	return wav, model.sample_rate

# huh
def decode_to_wave(resps: Tensor, device="cuda", levels=cfg.model.max_levels):
	return decode(resps, device=device, levels=levels)

def decode_to_file(resps: Tensor, path: Path, device="cuda"):
	wavs, sr = decode(resps, device=device)

	torchaudio.save(str(path), wavs.cpu(), sr)
	return wavs, sr

def _replace_file_extension(path, suffix):
	return (path.parent / path.name.split(".")[0]).with_suffix(suffix)

# an experimental way to include "trained" embeddings from the audio backend itself
# > b-but why not just initialize the embedding weights to these instead of fetching them at r-runtime
# each audio backend does their "embeddings" a different way that isn't just a embedding weights
#
# this is overkill and I don't feel like this benefits anything, but it was an idea I had
# this only really works if the embedding dims match, and either a Linear to rescale would be needed or semi-erroneously just padding with 0s
@torch.inference_mode()
def encode_as_embedding(codes: Tensor, quant_level: int = 0, sums=False, device="cuda"):
	model = _load_model(device)

	codes = codes.to(device=device, dtype=torch.int32)

	# yucky kludge
	if sums:
		if codes.dim() == 1:
			codes = rearrange(codes, "t -> t 1")

		if cfg.audio_backend == "dac":
			x = []
			for i in range(quant_level+1):
				emb = model.quantizer.quantizers[i]
				code = rearrange(codes[:, quant_level], "t -> 1 t")

				xi = emb.decode_code(code)
				xi = emb.out_proj(xi)
				x.append( xi[0].t() )

			return sum(x).detach()

		raise Exception(f'Currently only DAC is supported')


	if codes.dim() == 2:
		codes = codes[:, quant_level]

	codes = rearrange(codes, "t -> 1 t")

	# dac conveniently has its dim = 1024
	if cfg.audio_backend == "dac":
		emb = model.quantizer.quantizers[quant_level]

		x = emb.decode_code(codes)
		x = emb.out_proj(x)
		x = x[0].t().detach()

		return x

	"""
	# vocos inconveniently has its dim = 128
	elif cfg.audio_backend == "vocos":
		x = model.codes_to_features(codes)
	# encodec inconveniently has its dim = 300
	elif cfg.audio_backend == "encodec":
		...
	"""

	raise Exception(f'Currently only DAC is supported')

@torch.inference_mode()
def encode(wav: Tensor, sr: int = cfg.sample_rate, device="cuda", levels=cfg.model.max_levels, return_metadata=True):

	# DAC uses a different pathway
	if cfg.audio_backend == "dac":
		model = _load_dac_model(device, levels=levels )
		signal = AudioSignal(wav, sample_rate=sr)
		
		if not isinstance(levels, int):
			levels = 8 if model.model_type == "24khz" else None

		with torch.autocast("cuda", dtype=cfg.inference.dtype, enabled=cfg.inference.amp):
			artifact = model.compress(signal, win_duration=None, verbose=False, n_quantizers=levels)
		return artifact.codes if not return_metadata else artifact

	# AudioDec uses a different pathway
	if cfg.audio_backend == "audiodec":
		model = _load_audiodec_model(device, levels=levels )
		wav = wav.unsqueeze(0)
		wav = convert_audio(wav, sr, model.sample_rate, 1)
		wav = wav.to(device)

		# wav = rearrange(wav, "t c -> t 1 c").to(device)
		encoded = model.tx_encoder.encode(wav)
		quantized = model.tx_encoder.quantize(encoded)
		return quantized

	# vocos does not encode wavs to encodecs, so just use normal encodec
	model = _load_encodec_model(device, levels=levels)
	wav = wav.unsqueeze(0)
	wav = convert_audio(wav, sr, model.sample_rate, model.channels)
	wav = wav.to(device)

	with torch.autocast("cuda", dtype=cfg.inference.dtype, enabled=cfg.inference.amp):
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
# trims from the start, up to `target`
def trim( qnt, target, reencode=False ):
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
	
	wav = decode(qnt)[0]
	return encode(wav[start:end], cfg.sample_rate)[0].t()

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
def concat_audio( *args, reencode=False, device="cuda", levels=cfg.model.max_levels ):
	qnts = [ *args ]
	qnts = [ qnt for qnt in qnts if qnt is not None ]
	# just naively combine the codes
	if not reencode:
		return torch.concat( qnts )

	decoded = [ decode(qnt, device=device, levels=levels)[0] for qnt in qnts ]
	combined = torch.concat( decoded )
	return encode(combined, cfg.sample_rate, device=device, levels=levels)[0].t()

# merges two quantized audios together
# requires re-encoding because there's no good way to combine the waveforms of two audios without relying on some embedding magic
def merge_audio( *args, device="cuda", scale=[], levels=cfg.model.max_levels ):
	qnts = [ *args ]
	qnts = [ qnt for qnt in qnts if qnt is not None ]
	decoded = [ decode(qnt, device=device, levels=levels)[0] for qnt in qnts ]

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
	return encode(combined, cfg.sample_rate, device=device, levels=levels)[0].t()

"""
if __name__ == "__main__":
	cfg.sample_rate = 48_000
	cfg.audio_backend = "audiodec"

	wav, sr = torchaudio.load("in.wav")
	codes = encode( wav, sr ).t() # for some reason
	print( "ENCODED:", codes.shape, codes )
	wav, sr = decode( codes )
	print( "DECODED:", wav.shape, wav )
	torchaudio.save("out.wav", wav.cpu(), sr)
"""