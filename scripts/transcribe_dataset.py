import os
import json
import torch
import torchaudio
import whisperx

from tqdm.auto import tqdm
from pathlib import Path

# to-do: use argparser
batch_size = 16
device = "cuda" 
dtype = "float16"
model_name = "large-v3"

input_audio = "voices"
output_dataset = "training/metadata"

skip_existing = True
diarize = False

# 
model = whisperx.load_model(model_name, device, compute_type=dtype)
align_model, align_model_metadata, align_model_language = (None, None, None)
if diarize:
	diarize_model = whisperx.DiarizationPipeline(device=device)
else:
	diarize_model = None

def pad(num, zeroes):
	return str(num).zfill(zeroes+1)

for dataset_name in os.listdir(f'./{input_audio}/'):
	if not os.path.isdir(f'./{input_audio}/{dataset_name}/'):
		continue

	for speaker_id in tqdm(os.listdir(f'./{input_audio}/{dataset_name}/'), desc="Processing speaker"):
		if not os.path.isdir(f'./{input_audio}/{dataset_name}/{speaker_id}'):
			continue

		outpath = Path(f'./{output_dataset}/{dataset_name}/{speaker_id}/whisper.json')

		if outpath.exists():
			metadata = json.loads(open(outpath, 'r', encoding='utf-8').read())
		else:
			os.makedirs(f'./{output_dataset}/{dataset_name}/{speaker_id}/', exist_ok=True)
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

			if language[:2] not in ["ja"]:
				language = "en"

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