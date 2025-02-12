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

from vall_e.emb.process import pad, load_audio, process_items, process_jobs


def process(
	audio_backend="encodec",
	input_audio="LibriTTS_R",
	output_dataset="training",
	raise_exceptions=False,
	stride=0,
	stride_offset=0,
	slice="auto",
	batch_size=1,
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

	# Layout: ./LibriTTS_R/train-clean-100/103/1241
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

			if f'{group_name}/{speaker_id}' not in dataset:
				dataset.append(f'{group_name}/{speaker_id}')

			txts = []
			wavs = []

			for book_id in os.listdir(f'./{input_audio}/{group_name}/{speaker_id}'):
				if not os.path.isdir(f'./{input_audio}/{group_name}/{speaker_id}/{book_id}'):
					print("Is not dir:", f'./{input_audio}/{group_name}/{speaker_id}/{book_id}')
					continue

				for filename in os.listdir(f'./{input_audio}/{group_name}/{speaker_id}/{book_id}'):
					if ".wav" not in filename:
						continue

					inpath = Path(f'./{input_audio}/{group_name}/{speaker_id}/{book_id}/{filename}')
					textpath = _replace_file_extension(inpath, ".original.txt")
					if not inpath.exists() or not textpath.exists():
						missing["audio"].append(str(inpath))
						continue
				
					extension = os.path.splitext(filename)[-1][1:]
					fname = filename.replace(f'.{extension}', "")

					waveform, sample_rate = None, None
					language = "en"

					outpath = Path(f'./{output_dataset}/{group_name}/{speaker_id}/{fname}.{extension}')
					text = open(textpath, "r", encoding="utf-8").read()

					if len(text) == 0:
						continue

					if _replace_file_extension(outpath, audio_extension).exists():
						continue

					if waveform is None:
						waveform, sample_rate = load_audio(inpath)

				jobs.append(( outpath, waveform, sample_rate, text, language ))

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
	parser.add_argument("--dtype", type=str, default="bfloat16")
	parser.add_argument("--amp", action="store_true")
	parser.add_argument("--input-audio", type=str, default="LibriTTS_R")
	parser.add_argument("--output-dataset", type=str, default="training/dataset")
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--raise-exceptions", action="store_true")
	parser.add_argument("--stride", type=int, default=0)
	parser.add_argument("--stride-offset", type=int, default=0)
	parser.add_argument("--slice", type=str, default="auto")
	parser.add_argument("--low-memory", action="store_true")
	parser.add_argument("--batch-size", type=int, default=0)
	
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
		batch_size=args.batch_size,
		low_memory=args.low_memory,

		device=args.device,
		dtype=args.dtype,
		amp=args.amp,
	)

if __name__ == "__main__":
	main()