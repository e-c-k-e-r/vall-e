import torch

from dac import DACFile
from audiotools import AudioSignal
from dac.utils import load_model as load_dac_model

from typing import Union
from pathlib import Path
"""
Patch decode to skip things related to the metadata (namely the waveform trimming)
So far it seems the raw waveform can just be returned without any post-processing
A smart implementation would just reuse the values from the input prompt
"""
from dac.model.base import CodecMixin

@torch.no_grad()
def CodecMixin_compress(
	self,
	audio_path_or_signal: Union[str, Path, AudioSignal],
	win_duration: float = 1.0,
	verbose: bool = False,
	normalize_db: float = -16,
	n_quantizers: int = None,
) -> DACFile:
	"""Processes an audio signal from a file or AudioSignal object into
	discrete codes. This function processes the signal in short windows,
	using constant GPU memory.

	Parameters
	----------
	audio_path_or_signal : Union[str, Path, AudioSignal]
		audio signal to reconstruct
	win_duration : float, optional
		window duration in seconds, by default 5.0
	verbose : bool, optional
		by default False
	normalize_db : float, optional
		normalize db, by default -16

	Returns
	-------
	DACFile
		Object containing compressed codes and metadata
		required for decompression
	"""
	audio_signal = audio_path_or_signal
	if isinstance(audio_signal, (str, Path)):
		audio_signal = AudioSignal.load_from_file_with_ffmpeg(str(audio_signal))

	self.eval()
	original_padding = self.padding
	original_device = audio_signal.device

	audio_signal = audio_signal.clone()
	original_sr = audio_signal.sample_rate

	resample_fn = audio_signal.resample
	loudness_fn = audio_signal.loudness

	# If audio is > 10 minutes long, use the ffmpeg versions
	if audio_signal.signal_duration >= 10 * 60 * 60:
		resample_fn = audio_signal.ffmpeg_resample
		loudness_fn = audio_signal.ffmpeg_loudness

	original_length = audio_signal.signal_length
	resample_fn(self.sample_rate)
	input_db = loudness_fn()

	if normalize_db is not None:
		audio_signal.normalize(normalize_db)
	audio_signal.ensure_max_of_audio()

	nb, nac, nt = audio_signal.audio_data.shape
	audio_signal.audio_data = audio_signal.audio_data.reshape(nb * nac, 1, nt)
	win_duration = (
		audio_signal.signal_duration if win_duration is None else win_duration
	)

	if audio_signal.signal_duration <= win_duration:
		# Unchunked compression (used if signal length < win duration)
		self.padding = True
		n_samples = nt
		hop = nt
	else:
		# Chunked inference
		self.padding = False
		# Zero-pad signal on either side by the delay
		audio_signal.zero_pad(self.delay, self.delay)
		n_samples = int(win_duration * self.sample_rate)
		# Round n_samples to nearest hop length multiple
		n_samples = int(math.ceil(n_samples / self.hop_length) * self.hop_length)
		hop = self.get_output_length(n_samples)

	codes = []
	range_fn = range if not verbose else tqdm.trange

	for i in range_fn(0, nt, hop):
		x = audio_signal[..., i : i + n_samples]
		x = x.zero_pad(0, max(0, n_samples - x.shape[-1]))

		audio_data = x.audio_data.to(self.device)
		audio_data = self.preprocess(audio_data, self.sample_rate)
		with torch.autocast("cuda", dtype=cfg.inference.dtype, enabled=cfg.inference.amp):
			_, c, _, _, _ = self.encode(audio_data, n_quantizers)
		codes.append(c.to(original_device))
		chunk_length = c.shape[-1]

	codes = torch.cat(codes, dim=-1)

	dac_file = DACFile(
		codes=codes,
		chunk_length=chunk_length,
		original_length=original_length,
		input_db=input_db,
		channels=nac,
		sample_rate=original_sr,
		padding=self.padding,
		dac_version="1.0.0",
		#dac_version=SUPPORTED_VERSIONS[-1],
	)

	if n_quantizers is not None:
		codes = codes[:, :n_quantizers, :]

	self.padding = original_padding
	return dac_file

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

CodecMixin.compress = CodecMixin_compress
CodecMixin.decompress = CodecMixin_decompress