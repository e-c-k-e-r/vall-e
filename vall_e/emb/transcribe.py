"""
# Handles transcribing audio provided through --input-audio
"""

import os
import json
import argparse

import torch
import torchaudio

import whisperx

from tqdm.auto import tqdm
from pathlib import Path

def pad(num, zeroes):
	return str(num).zfill(zeroes+1)

def process_items( items, stride=0, stride_offset=0 ):
	items = sorted( items )
	return items if stride == 0 else [ item for i, item in enumerate( items ) if (i+stride_offset) % stride == 0 ]

def transcribe(
	input_audio = "voices",
	input_voice = None,
	output_metadata = "training/metadata",
	model_name = "large-v3",
	
	skip_existing = True,
	diarize = False,

	stride = 0,
	stride_offset = 0,

	batch_size = 16,
	device = "cuda",
	dtype = "float16",
):
	# to-do: make this also prepared from args
	language_map = {} # k = group, v = language

	ignore_groups = [] # skip these groups
	ignore_speakers = [] # skip these speakers

	only_groups = [] # only process these groups
	only_speakers = [] # only process these speakers

	if input_voice is not None:
		only_speakers = [input_voice]

	# 
	model = whisperx.load_model(model_name, device, compute_type=dtype)
	align_model, align_model_metadata, align_model_language = (None, None, None)
	if diarize:
		diarize_model = whisperx.DiarizationPipeline(device=device)
	else:
		diarize_model = None


	for dataset_name in os.listdir(f'./{input_audio}/'):
		if not os.path.isdir(f'./{input_audio}/{dataset_name}/'):
			continue

		if group_name in ignore_groups:
			continue
		if only_groups and group_name not in only_groups:
			continue

		for speaker_id in tqdm(process_items(os.listdir(f'./{input_audio}/{dataset_name}/')), desc="Processing speaker"):
			if not os.path.isdir(f'./{input_audio}/{dataset_name}/{speaker_id}'):
				continue

			if speaker_id in ignore_speakers:
				continue
			if only_speakers and speaker_id not in only_speakers:
				continue

			outpath = Path(f'./{output_metadata}/{dataset_name}/{speaker_id}/whisper.json')

			if outpath.exists():
				metadata = json.loads(open(outpath, 'r', encoding='utf-8').read())
			else:
				os.makedirs(f'./{output_metadata}/{dataset_name}/{speaker_id}/', exist_ok=True)
				metadata = {}

			for filename in tqdm(os.listdir(f'./{input_audio}/{dataset_name}/{speaker_id}/'), desc=f"Processing speaker: {speaker_id}"):
				if skip_existing and filename in metadata:
					continue

				if ".json" in filename:
					continue

				inpath = f'./{input_audio}/{dataset_name}/{speaker_id}/{filename}'

				if os.path.isdir(inpath):
					continue
				
				metadata[filename] = {
					"segments": [],
					"language": "",
					"text": "",
					"start": 0,
					"end": 0,
				}

				audio = whisperx.load_audio(inpath)
				result = model.transcribe(audio, batch_size=batch_size)
				language = result["language"]

				"""
				if language[:2] not in ["ja"]:
					language = "en"
				"""

				if align_model_language != language:
					tqdm.write(f'Loading language: {language}')
					align_model, align_model_metadata = whisperx.load_align_model(language_code=language, device=device)
					align_model_language = language

				result = whisperx.align(result["segments"], align_model, align_model_metadata, audio, device, return_char_alignments=False)

				metadata[filename]["segments"] = result["segments"]
				metadata[filename]["language"] = language

				if diarize_model is not None:
					diarize_segments = diarize_model(audio)
					result = whisperx.assign_word_speakers(diarize_segments, result)

				text = []
				start = 0
				end = 0
				for segment in result["segments"]:
					text.append( segment["text"] )
					start = min( start, segment["start"] )
					end = max( end, segment["end"] )

				metadata[filename]["text"] = " ".join(text).strip()
				metadata[filename]["start"] = start
				metadata[filename]["end"] = end

				open(outpath, 'w', encoding='utf-8').write(json.dumps(metadata))

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--input-audio", type=str, default="voices")
	parser.add_argument("--input-voice", type=str, default=None)
	parser.add_argument("--output-metadata", type=str, default="training/metadata")

	parser.add_argument("--model-name", type=str, default="large-v3")
	parser.add_argument("--skip-existing", action="store_true")
	parser.add_argument("--diarize", action="store_true")
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--stride", type=int, default=0)
	parser.add_argument("--stride-offset", type=int, default=0)

	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--dtype", type=str, default="bfloat16")
	parser.add_argument("--amp", action="store_true")
	# parser.add_argument("--raise-exceptions", action="store_true")

	args = parser.parse_args()
	
	# do some assumption magic
	# to-do: find a nice way to spawn multiple processes where tqdm plays nicely
	if args.device.isnumeric():
		args.stride = torch.cuda.device_count()
		args.stride_offset = int(args.device)
		args.device = f'cuda:{args.device}'

	transcribe(
		input_audio = args.input_audio,
		input_voice = args.input_voice,
		output_metadata = args.output_metadata,
		model_name = args.model_name,

		skip_existing = args.skip_existing,
		diarize = args.diarize,

		stride = args.stride,
		stride_offset = args.stride_offset,

		batch_size = args.batch_size,
		device = args.device,
		dtype = args.dtype,
	)

if __name__ == "__main__":
	main()