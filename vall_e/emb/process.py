"""
# Handles processing audio provided through --input-audio of adequately annotated transcriptions provided through --input-metadata (through transcribe.py)
# Outputs NumPy objects containing quantized audio and adequate metadata for use of loading in the trainer through --output-dataset
"""

import os
import json
import argparse
import torch
import torchaudio
import numpy as np
import logging

_logger = logging.getLogger(__name__)

from tqdm.auto import tqdm
from pathlib import Path

from ..config import cfg

# need to validate if this is safe to import before modifying the config
from .g2p import encode as phonemize
from .qnt import encode as quantize, encode_batch as quantize_batch

def pad(num, zeroes):
	return str(num).zfill(zeroes+1)

def load_audio( path, device=None, dtype=None ):
	waveform, sr = torchaudio.load( path )
	if waveform.shape[0] > 1:
		# mix channels
		waveform = torch.mean(waveform, dim=0, keepdim=True)
	if dtype is not None:
		waveform = waveform.to(dtype)
	if device is not None:
		waveform = waveform.to(device)
	return waveform, sr

def process_items( items, stride=0, stride_offset=0 ):
	items = sorted( items )
	return items if stride == 0 else [ item for i, item in enumerate( items ) if (i+stride_offset) % stride == 0 ]

def process_job( outpath, waveform, sample_rate, text=None, language="en", device="cuda", dtype=None ):
	# encodec requires this to be on CPU for resampling
	qnt = quantize(waveform, sr=sample_rate, device=device, dtype=dtype)

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

	if text:
		text = text.strip()
		state_dict['metadata'] |= {
			"text": text,
			"phonemes": phonemize(text, language=language),
			"language": language,
		}
	
	np.save(open(outpath, "wb"), state_dict)

def process_batched_jobs( jobs, speaker_id="", device=None, raise_exceptions=True, batch_size=1, dtype=None ):
	if not jobs:
		return

	# sort to avoid egregious padding
	jobs = sorted(jobs, key=lambda x: x[1].shape[-1], reverse=True)

	buffer = []
	batches = []

	for job in jobs:
		buffer.append(job)
		if len(buffer) >= batch_size:
			batches.append(buffer)
			buffer = []
	
	if buffer:
		batches.append(buffer)
		buffer = []

	for batch in tqdm(batches, desc=f'Quantizing {speaker_id} (batch size: {batch_size})'):
		wavs = []
		srs = []
		
		for outpath, waveform, sample_rate, text, language in batch:
			wavs.append(waveform)
			srs.append(sample_rate)
		
		try:
			codes = quantize_batch(wavs, sr=srs, device=device, dtype=dtype)
		except Exception as e:
			_logger.error(f"Failed to quantize: {outpath}: {str(e)}")
			if raise_exceptions:
				raise e
			continue

		for (outpath, waveform, sample_rate, text, language), qnt in zip( batch, codes ):
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

			if text:
				text = text.strip()
				state_dict['metadata'] |= {
					"text": text,
					"phonemes": phonemize(text, language=language),
					"language": language,
				}
			
			np.save(open(outpath, "wb"), state_dict)

def process_jobs( jobs, speaker_id="", device=None, raise_exceptions=True, batch_size=1, dtype=None ):
	if not jobs:
		return

	# batch things
	if batch_size > 1:
		return process_batched_jobs( jobs, speaker_id=speaker_id, device=device, raise_exceptions=raise_exceptions, batch_size=batch_size, dtype=dtype )
	
	for job in tqdm(jobs, desc=f"Quantizing: {speaker_id}"):
		outpath, waveform, sample_rate, text, language = job
		try:
			process_job( outpath, waveform, sample_rate, text, language, device, dtype=dtype )
		except Exception as e:
			_logger.error(f"Failed to quantize: {outpath}: {str(e)}")
			if raise_exceptions:
				raise e
			continue

def process(
	audio_backend="encodec",
	input_audio="voices",
	input_voice=None,
	input_metadata="metadata",
	output_dataset="training",
	raise_exceptions=False,
	stride=0,
	stride_offset=0,
	slice="auto",
	batch_size=1,
	max_duration=None,
	skip_existing_folders=False,
	low_memory=False,

	device="cuda",
	dtype="float16",
	amp=False,
):
	# prepare from args
	cfg.device = device
	cfg.set_audio_backend(audio_backend)
	audio_extension = cfg.audio_backend_extension

	cfg.inference.weight_dtype = dtype # "bfloat16"
	cfg.inference.amp = amp # False

	dtype = cfg.inference.dtype if not amp else None

	output_dataset = f"{output_dataset}/{'2' if cfg.sample_rate == 24_000 else '4'}{'8' if cfg.sample_rate == 48_000 else '4'}KHz-{cfg.audio_backend}" # "training"

	# to-do: make this also prepared from args
	language_map = {} # k = group, v = language

	ignore_groups = [] # skip these groups
	ignore_speakers = [] # skip these speakers

	only_groups = [] # only process these groups
	only_speakers = [] # only process these speakers

	always_slice_groups = ["Audiobooks", "LibriVox"] # always slice from this group
	audio_only = ["Noise"] # special pathway for processing audio only (without a transcription)

	missing = {
		"transcription": [],
		"audio": []
	}
	dataset = []

	if input_voice is not None:
		only_speakers = [input_voice]

	for group_name in sorted(os.listdir(f'./{input_audio}/')):
		if not os.path.isdir(f'./{input_audio}/{group_name}/'):
			_logger.warning(f'Is not dir:" /{input_audio}/{group_name}/')
			continue

		if group_name in ignore_groups:
			continue
		if only_groups and group_name not in only_groups:
			continue

		for speaker_id in tqdm(process_items(os.listdir(f'./{input_audio}/{group_name}/'), stride=stride, stride_offset=stride_offset), desc=f"Processing speaker in {group_name}"):
			if not os.path.isdir(f'./{input_audio}/{group_name}/{speaker_id}'):
				_logger.warning(f'Is not dir: ./{input_audio}/{group_name}/{speaker_id}')
				continue
			
			if speaker_id in ignore_speakers:
				continue
			if only_speakers and speaker_id not in only_speakers:
				continue
			
			outfolder = Path(f'./{output_dataset}/{group_name}/{speaker_id}/')

			if skip_existing_folders and outfolder.exists():
				continue
			
			outfolder.mkdir(parents=True, exist_ok=True)

			if speaker_id in audio_only:
				for filename in sorted(os.listdir(f'./{input_audio}/{group_name}/{speaker_id}/')):
					inpath = Path(f'./{input_audio}/{group_name}/{speaker_id}/{filename}')
					outpath = Path(f'./{output_dataset}/{group_name}/{speaker_id}/{filename}').with_suffix(audio_extension)

					if outpath.exists():
						continue

					waveform, sample_rate = load_audio( inpath, dtype=dtype )
					qnt = quantize(waveform, sr=sample_rate, device=device)

					process_job(outpath, waveform, sample_rate)

				continue
			
			metadata_path = Path(f'./{input_metadata}/{group_name}/{speaker_id}/whisper.json')
			if not metadata_path.exists():
				missing["transcription"].append(str(metadata_path))
				_logger.warning(f'Missing transcription metadata: ./{input_audio}/{group_name}/{speaker_id}/whisper.json')
				continue

			try:
				metadata = json.loads(open(metadata_path, "r", encoding="utf-8").read())
			except Exception as e:
				missing["transcription"].append(str(metadata_path))
				_logger.warning(f'Failed to open transcription metadata: ./{input_audio}/{group_name}/{speaker_id}/whisper.json: {e}')
				continue

			if f'{group_name}/{speaker_id}' not in dataset:
				dataset.append(f'{group_name}/{speaker_id}')

			jobs = []

			use_slices = slice == True or (slice == "auto" and len(metadata.keys()) == 1) or group_name in always_slice_groups

			for filename in sorted(metadata.keys()):
				inpath = Path(f'./{input_audio}/{group_name}/{speaker_id}/{filename}')

				"""
				if not inpath.exists():
					missing["audio"].append(str(inpath))
					continue
				"""

				extension = os.path.splitext(filename)[-1][1:]
				fname = filename.replace(f'.{extension}', "")

				waveform, sample_rate = None, None
				language = language_map[group_name] if group_name in language_map else (metadata[filename]["language"] if "language" in metadata[filename] else "en")

				if len(metadata[filename]["segments"]) == 0 or not use_slices:
					outpath = Path(f'./{output_dataset}/{group_name}/{speaker_id}/{fname}.{extension}').with_suffix(audio_extension)
					text = metadata[filename]["text"]

					if len(text) == 0 or outpath.exists():
						continue

					# audio not already loaded, load it
					if waveform is None:
						waveform, sample_rate = load_audio( inpath, dtype=dtype )

					if max_duration and waveform.shape[-1] / sample_rate > max_duration:
						continue

					jobs.append(( outpath, waveform, sample_rate, text, language ))
				else:
					i = 0
					presliced = not inpath.exists()
					
					for segment in metadata[filename]["segments"]:
						id = pad(i, 4)
						i = i + 1

						if presliced:
							inpath = Path(f'./{input_audio}/{group_name}/{speaker_id}/{fname}_{id}.{extension}')

						if not inpath.exists():
							missing["audio"].append(str(inpath))
							continue

						outpath = Path(f'./{output_dataset}/{group_name}/{speaker_id}/{fname}_{id}.{extension}').with_suffix(audio_extension)
						text = segment["text"]

						if len(text) == 0 or outpath.exists():
							continue

						# audio not already loaded, load it
						if waveform is None:
							waveform, sample_rate = load_audio( inpath, dtype=dtype )

						start = int((segment['start']-0.05) * sample_rate)
						end = int((segment['end']+0.5) * sample_rate)

						if max_duration and (end - start) / sample_rate > max_duration:
							continue

						if not presliced:
							if start < 0:
								start = 0
							if end >= waveform.shape[-1]:
								end = waveform.shape[-1] - 1

							if end - start < 0:
								continue

						jobs.append(( outpath, waveform if presliced else waveform[:, start:end], sample_rate, text, language ))

				# processes audio files one at a time
				if low_memory:
					process_jobs( jobs, device=device, speaker_id=f'{speaker_id}/{filename}', raise_exceptions=raise_exceptions, batch_size=batch_size, dtype=dtype if not amp else None )
					jobs = []
			
			# processes all audio files for a given speaker
			if not low_memory:
				process_jobs( jobs, device=device, speaker_id=speaker_id, raise_exceptions=raise_exceptions, batch_size=batch_size, dtype=dtype if not amp else None )
				jobs = []

	open(f"./{output_dataset}/missing.json", 'w', encoding='utf-8').write(json.dumps(missing))
	open(f"./{output_dataset}/dataset.json", 'w', encoding='utf-8').write(json.dumps(dataset))

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--audio-backend", type=str, default="encodec")
	parser.add_argument("--input-audio", type=str, default="voices")
	parser.add_argument("--input-voice", type=str, default=None)
	parser.add_argument("--input-metadata", type=str, default="training/metadata")
	parser.add_argument("--output-dataset", type=str, default="training/dataset")
	parser.add_argument("--raise-exceptions", action="store_true")
	parser.add_argument("--low-memory", action="store_true")
	parser.add_argument("--skip-existing-folders", action="store_true")
	parser.add_argument("--stride", type=int, default=0)
	parser.add_argument("--stride-offset", type=int, default=0)
	parser.add_argument("--slice", type=str, default="auto")
	parser.add_argument("--batch-size", type=int, default=0)
	parser.add_argument("--max-duration", type=int, default=0)
	
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--dtype", type=str, default="bfloat16")
	parser.add_argument("--amp", action="store_true")
	
	args = parser.parse_args()

	# do some assumption magic
	# to-do: find a nice way to spawn multiple processes where tqdm plays nicely
	if args.device.isnumeric():
		args.stride = torch.cuda.device_count()
		args.stride_offset = int(args.device)
		args.device = f'cuda:{args.device}'

	if args.slice == "true":
		args.slice = True
	elif args.slice == "false":
		args.slice = False

	process(
		audio_backend=args.audio_backend,
		input_audio=args.input_audio,
		input_voice=args.input_voice,
		input_metadata=args.input_metadata,
		output_dataset=args.output_dataset,
		raise_exceptions=args.raise_exceptions,
		stride=args.stride,
		stride_offset=args.stride_offset,
		slice=args.slice,
		batch_size=args.batch_size,
		max_duration=args.max_duration,
		skip_existing_folders=args.skip_existing_folders,
		
		low_memory=args.low_memory,

		device=args.device,
		dtype=args.dtype,
		amp=args.amp,
	)

if __name__ == "__main__":
	main()