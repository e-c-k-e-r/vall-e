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

from tqdm.auto import tqdm
from pathlib import Path

from ..config import cfg

# need to validate if this is safe to import before modifying the config
from .g2p import encode as phonemize
from .qnt import encode as quantize, _replace_file_extension

def pad(num, zeroes):
	return str(num).zfill(zeroes+1)

def load_audio( path, device ):
	waveform, sr = torchaudio.load( path )
	if waveform.shape[0] > 1:
		# mix channels
		waveform = torch.mean(waveform, dim=0, keepdim=True)
	return waveform.to(device=device), sr

def process_items( items, stride=0, stride_offset=0 ):
	items = sorted( items )
	return items if stride == 0 else [ item for i, item in enumerate( items ) if (i+stride_offset) % stride == 0 ]

def process_job( outpath, text, language, waveform, sample_rate ):
	phones = phonemize(text, language=language)
	qnt = quantize(waveform, sr=sample_rate, device=waveform.device)

	if cfg.audio_backend == "dac":
		np.save(open(outpath, "wb"), {
			"codes": qnt.codes.cpu().numpy().astype(np.uint16),
			"metadata": {
				"original_length": qnt.original_length,
				"sample_rate": qnt.sample_rate,
				
				"input_db": qnt.input_db.cpu().numpy().astype(np.float32),
				"chunk_length": qnt.chunk_length,
				"channels": qnt.channels,
				"padding": qnt.padding,
				"dac_version": "1.0.0",

				"text": text.strip(),
				"phonemes": "".join(phones),
				"language": language,
			},
		})
	else:						
		np.save(open(outpath, "wb"), {
			"codes": qnt.cpu().numpy().astype(np.uint16),
			"metadata": {
				"original_length": waveform.shape[-1],
				"sample_rate": sample_rate,

				"text": text.strip(),
				"phonemes": "".join(phones),
				"language": language,
			},
		})

def process_jobs( jobs, speaker_id="", raise_exceptions=True ):
	if not jobs:
		return
	
	for job in tqdm(jobs, desc=f"Quantizing: {speaker_id}"):
		outpath, text, language, waveform, sample_rate = job
		try:
			process_job( outpath, text, language, waveform, sample_rate  )
		except Exception as e:
			print(f"Failed to quantize: {outpath}:", e)
			if raise_exceptions:
				raise e
			continue

def process(
		audio_backend="encodec",
		input_audio="voices",
		input_metadata="metadata",
		output_dataset="training",
		raise_exceptions=False,
		stride=0,
		stride_offset=0,
		slice="auto",

		low_memory=False,

		device="cuda",
		dtype="float16",
		amp=False,
	):
	# encodec / vocos

	if audio_backend in ["encodec", "vocos"]:
		audio_extension = ".enc"
		cfg.sample_rate = 24_000
		cfg.model.resp_levels = 8
	elif audio_backend == "dac":
		audio_extension = ".dac"
		cfg.sample_rate = 44_100
		cfg.model.resp_levels = 9
	elif cfg.audio_backend == "audiodec":
		sample_rate = 48_000
		audio_extension = ".dec"
		cfg.model.resp_levels = 8 # ?
	else:
		raise Exception(f"Unknown audio backend: {audio_backend}")

	# prepare from args
	cfg.audio_backend = audio_backend # "encodec"
	cfg.inference.weight_dtype = dtype # "bfloat16"
	cfg.inference.amp = amp # False

	output_dataset = f"{output_dataset}/{'2' if cfg.sample_rate == 24_000 else '4'}{'8' if cfg.sample_rate == 48_000 else '4'}KHz-{cfg.audio_backend}" # "training"

	language_map = {} # k = group, v = language

	ignore_groups = [] # skip these groups
	ignore_speakers = [] # skip these speakers

	only_groups = [] # only process these groups
	only_speakers = [] # only process these speakers

	always_slice_groups = [] # always slice from this group
	audio_only = ["Noise"] # special pathway for processing audio only (without a transcription)

	missing = {
		"transcription": [],
		"audio": []
	}
	dataset = []

	for group_name in sorted(os.listdir(f'./{input_audio}/')):
		if not os.path.isdir(f'./{input_audio}/{group_name}/'):
			print("Is not dir:", f'./{input_audio}/{group_name}/')
			continue

		if group_name in ignore_groups:
			continue
		if only_groups and group_name not in only_groups:
			continue

		for speaker_id in tqdm(process_items(os.listdir(f'./{input_audio}/{group_name}/'), stride=stride, stride_offset=stride_offset), desc=f"Processing speaker in {group_name}"):
			if not os.path.isdir(f'./{input_audio}/{group_name}/{speaker_id}'):
				print("Is not dir:", f'./{input_audio}/{group_name}/{speaker_id}')
				continue
			
			if speaker_id in ignore_speakers:
				continue
			if only_speakers and speaker_id not in only_speakers:
				continue

			os.makedirs(f'./{output_dataset}/{group_name}/{speaker_id}/', exist_ok=True)

			if speaker_id in audio_only:
				for filename in sorted(os.listdir(f'./{input_audio}/{group_name}/{speaker_id}/')):
					inpath = Path(f'./{input_audio}/{group_name}/{speaker_id}/{filename}')
					outpath = Path(f'./{output_dataset}/{group_name}/{speaker_id}/{filename}')
					outpath = _replace_file_extension(outpath, audio_extension)

					if outpath.exists():
						continue

					waveform, sample_rate = load_audio( inpath, device )
					qnt = quantize(waveform, sr=sample_rate, device=device)

					if cfg.audio_backend == "dac":
						np.save(open(outpath, "wb"), {
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
						})
					else:
						np.save(open(outpath, "wb"), {
							"codes": qnt.cpu().numpy().astype(np.uint16),
							"metadata": {
								"original_length": waveform.shape[-1],
								"sample_rate": sample_rate,
							},
						})

				continue
			
			metadata_path = Path(f'./{input_metadata}/{group_name}/{speaker_id}/whisper.json')
			if not metadata_path.exists():
				missing["transcription"].append(str(metadata_path))
				continue

			try:
				metadata = json.loads(open(metadata_path, "r", encoding="utf-8").read())
			except Exception as e:
				missing["transcription"].append(str(metadata_path))
				continue

			if f'{group_name}/{speaker_id}' not in dataset:
				dataset.append(f'{group_name}/{speaker_id}')

			jobs = []

			use_slices = slice == True or (slice == "auto" and len(metadata.keys()) == 1) or group_name in always_slice_groups

			for filename in sorted(metadata.keys()):
				inpath = Path(f'./{input_audio}/{group_name}/{speaker_id}/{filename}')
				if not inpath.exists():
					missing["audio"].append(str(inpath))
					continue
				
				extension = os.path.splitext(filename)[-1][1:]
				fname = filename.replace(f'.{extension}', "")

				waveform, sample_rate = None, None
				language = language_map[group_name] if group_name in language_map else (metadata[filename]["language"] if "language" in metadata[filename] else "en")

				if len(metadata[filename]["segments"]) == 0 or not use_slices:
					outpath = Path(f'./{output_dataset}/{group_name}/{speaker_id}/{fname}.{extension}')
					outpath = _replace_file_extension(outpath, audio_extension)
					text = metadata[filename]["text"]

					if len(text) == 0 or outpath.exists():
						continue

					# audio not already loaded, load it
					if waveform is None:
						waveform, sample_rate = load_audio( inpath, device )

					jobs.append(( outpath, text, language, waveform, sample_rate ))
				else:
					i = 0
					for segment in metadata[filename]["segments"]:
						id = pad(i, 4)
						i = i + 1

						outpath = Path(f'./{output_dataset}/{group_name}/{speaker_id}/{fname}_{id}.{extension}')
						outpath = _replace_file_extension(outpath, audio_extension)
						text = segment["text"]

						if len(text) == 0 or outpath.exists():
							continue

						# audio not already loaded, load it
						if waveform is None:
							waveform, sample_rate = load_audio( inpath, device )

						start = int(segment['start'] * sample_rate)
						end = int(segment['end'] * sample_rate)

						if start < 0:
							start = 0
						if end >= waveform.shape[-1]:
							end = waveform.shape[-1] - 1

						if end - start < 0:
							continue

						jobs.append(( outpath, text, language, waveform[:, start:end], sample_rate ))

				# processes audio files one at a time
				if low_memory:
					process_jobs( jobs, speaker_id=f'{speaker_id}/{filename}', raise_exceptions=raise_exceptions )
					jobs = []
			
			# processes all audio files for a given speaker
			if not low_memory:
				process_jobs( jobs, speaker_id=speaker_id, raise_exceptions=raise_exceptions )
				jobs = []

	open(f"./{output_dataset}/missing.json", 'w', encoding='utf-8').write(json.dumps(missing))
	open(f"./{output_dataset}/dataset.json", 'w', encoding='utf-8').write(json.dumps(dataset))

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--audio-backend", type=str, default="encodec")
	parser.add_argument("--input-audio", type=str, default="voices")
	parser.add_argument("--input-metadata", type=str, default="training/metadata")
	parser.add_argument("--output-dataset", type=str, default="training/dataset")
	parser.add_argument("--raise-exceptions", action="store_true")
	parser.add_argument("--low-memory", action="store_true")
	parser.add_argument("--stride", type=int, default=0)
	parser.add_argument("--stride-offset", type=int, default=0)
	parser.add_argument("--slice", type=str, default="auto")
	
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

	process(
		audio_backend=args.audio_backend,
		input_audio=args.input_audio,
		input_metadata=args.input_metadata,
		output_dataset=args.output_dataset,
		raise_exceptions=args.raise_exceptions,
		stride=args.stride,
		stride_offset=args.stride_offset,
		slice=args.slice,
		
		low_memory=args.low_memory,

		device=args.device,
		dtype=args.dtype,
		amp=args.amp,
	)

if __name__ == "__main__":
	main()