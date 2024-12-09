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

from vall_e.config import cfg

from vall_e.emb.g2p import encode as phonemize
from vall_e.emb.qnt import encode as quantize, _replace_file_extension, convert_audio

def pad(num, zeroes):
	return str(num).zfill(zeroes+1)

def process_items( items, stride=0, stride_offset=0 ):
	items = sorted( items )
	return items if stride == 0 else [ item for i, item in enumerate( items ) if (i+stride_offset) % stride == 0 ]

def load_audio( path, device="cuda" ):
	waveform, sample_rate = torchaudio.load(path)
	if waveform.shape[0] > 1:
		waveform = torch.mean(waveform, dim=0, keepdim=True)
	waveform = convert_audio(waveform, sample_rate, cfg.sample_rate, 1)
	return waveform.to(device=device), cfg.sample_rate

def process(
		audio_backend="encodec",
		input_audio="Emilia",
		output_dataset="training",
		raise_exceptions=False,
		stride=0,
		stride_offset=0,
		slice="auto",

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

	missing = {
		"transcription": [],
		"audio": []
	}
	dataset = []

	# Layout: ./Emilia/JA/JA-B000000/JA_B00000_S00000_W000000.{json|mp3}
	for language in sorted(os.listdir(f'./{input_audio}/')):
		if not os.path.isdir(f'./{input_audio}/{language}/'):
			print("Is not dir:", f'./{input_audio}/{language}/')
			continue

		if language in ignore_groups:
			continue

		if only_groups and language not in only_groups:
			continue

		group_name = "Emilia"

		for speaker_group in tqdm(process_items(os.listdir(f'./{input_audio}/{language}/'), stride=stride, stride_offset=stride_offset), desc=f"Processing speaker in {language}"):
			if not os.path.isdir(f'./{input_audio}/{language}/{speaker_group}'):
				print("Is not dir:", f'./{input_audio}/{language}/{speaker_group}')
				continue
			
			if speaker_group in ignore_speakers:
				continue
			if only_speakers and speaker_group not in only_speakers:
				continue

			if f'{group_name}/{speaker_group}' not in dataset:
				dataset.append(f'{group_name}/{speaker_group}')

			txts = []
			wavs = []

			for filename in os.listdir(f'./{input_audio}/{language}/{speaker_group}'):
				if ".mp3" not in filename:
					continue

				inpath = Path(f'./{input_audio}/{language}/{speaker_group}/{filename}')
				jsonpath = _replace_file_extension(inpath, ".json")
				if not inpath.exists() or not jsonpath.exists():
					missing["audio"].append(str(inpath))
					continue
			
				extension = os.path.splitext(filename)[-1][1:]
				fname = filename.replace(f'.{extension}', "")

				waveform, sample_rate = None, None
				metadata = json.load(open(jsonpath, "r", encoding="utf-8"))
				if "text" not in metadata:
					continue
				speaker_id = metadata["speaker"]
				outpath = Path(f'./{output_dataset}/{group_name}/{speaker_group}/{speaker_id}/{fname}.{extension}')
				os.makedirs(f'./{output_dataset}/{group_name}/{speaker_group}/{speaker_id}/', exist_ok=True)

				if _replace_file_extension(outpath, audio_extension).exists():
					continue

				text = metadata["text"]

				if waveform is None:
					waveform, sample_rate = load_audio(inpath)

				wavs.append((
					outpath,
					text,
					language,
					waveform,
					sample_rate
				))

			if len(wavs) > 0:
				for job in tqdm(wavs, desc=f"Quantizing: {speaker_id}"):
					try:
						outpath, text, language, waveform, sample_rate = job

						phones = phonemize(text, language=f'{language}'.lower())
						qnt = quantize(waveform, sr=sample_rate, device=device)


						if cfg.audio_backend == "dac":
							np.save(open(_replace_file_extension(outpath, audio_extension), "wb"), {
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
							np.save(open(_replace_file_extension(outpath, audio_extension), "wb"), {
								"codes": qnt.cpu().numpy().astype(np.uint16),
								"metadata": {
									"original_length": waveform.shape[-1],
									"sample_rate": sample_rate,

									"text": text.strip(),
									"phonemes": "".join(phones),
									"language": language,
								},
							})
					except Exception as e:
						print(f"Failed to quantize: {outpath}:", e)
						if raise_exceptions:
							raise e
						continue

	open(f"./{output_dataset}/missing.json", 'w', encoding='utf-8').write(json.dumps(missing))
	open(f"./{output_dataset}/dataset.json", 'w', encoding='utf-8').write(json.dumps(dataset))

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--audio-backend", type=str, default="encodec")
	parser.add_argument("--dtype", type=str, default="bfloat16")
	parser.add_argument("--amp", action="store_true")
	parser.add_argument("--input-audio", type=str, default="Emilia")
	parser.add_argument("--output-dataset", type=str, default="training/dataset")
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--raise-exceptions", action="store_true")
	parser.add_argument("--stride", type=int, default=0)
	parser.add_argument("--stride-offset", type=int, default=0)
	parser.add_argument("--slice", type=str, default="auto")
	
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
		output_dataset=args.output_dataset,
		raise_exceptions=args.raise_exceptions,
		stride=args.stride,
		stride_offset=args.stride_offset,
		slice=args.slice,

		device=args.device,
		dtype=args.dtype,
		amp=args.amp,
	)

if __name__ == "__main__":
	main()