import os
import json
import torch
import torchaudio
import whisperx

from tqdm.auto import tqdm
from pathlib import Path

device = "cuda" 
batch_size = 16
dtype = "float16"
model_size = "large-v2"

input_audio = "voice"
output_dataset = "metadata"
skip_existing = True

model = whisperx.load_model(model_size, device, compute_type=dtype)

align_model, align_model_metadata, align_model_language = (None, None, None)

def pad(num, zeroes):
	return str(num).zfill(zeroes+1)

for dataset_name in os.listdir(f'./{input_audio}/'):
	if not os.path.isdir(f'./{input_audio}/{dataset_name}/'):
		print("Is not dir:", f'./{input_audio}/{dataset_name}/')
		continue

	for speaker_id in tqdm(os.listdir(f'./{input_audio}/{dataset_name}/'), desc="Processing speaker"):
		if not os.path.isdir(f'./{input_audio}/{dataset_name}/{speaker_id}'):
			print("Is not dir:", f'./{input_audio}/{dataset_name}/{speaker_id}')
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

			inpath = f'./{input_audio}/{dataset_name}/{speaker_id}/{filename}'
			
			metadata[filename] = {
				"segments": [],
				"language": "",
				"text": [],
			}

			audio = whisperx.load_audio(inpath)
			result = model.transcribe(audio, batch_size=batch_size)
			language = result["language"]

			if align_model_language != language:
				tqdm.write(f'Loading language: {language}')
				align_model, align_model_metadata = whisperx.load_align_model(language_code=language, device=device)
				align_model_language = language

			result = whisperx.align(result["segments"], align_model, align_model_metadata, audio, device, return_char_alignments=False)

			metadata[filename]["segments"] = result["segments"]
			metadata[filename]["language"] = language

			text = []
			for segment in result["segments"]:
				id = len(text)
				text.append( segment["text"] )
				metadata[filename]["segments"][id]["id"] = id

			metadata[filename]["text"] = " ".join(text).strip()

			open(outpath, 'w', encoding='utf-8').write(json.dumps(metadata))