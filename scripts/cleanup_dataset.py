import os
import json
import torch
import torchaudio

from tqdm.auto import tqdm
from pathlib import Path

input_dataset = "training/metadata"
output_dataset = "training/metadata-cleaned"

def pad(num, zeroes):
	return str(num).zfill(zeroes+1)

for dataset_name in os.listdir(f'./{input_dataset}/'):
	if not os.path.isdir(f'./{input_dataset}/{dataset_name}/'):
		print("Is not dir:", f'./{input_dataset}/{dataset_name}/')
		continue

	for speaker_id in tqdm(os.listdir(f'./{input_dataset}/{dataset_name}/'), desc=f"Processing speaker: {dataset_name}"):
		if not os.path.isdir(f'./{input_dataset}/{dataset_name}/{speaker_id}'):
			print("Is not dir:", f'./{input_dataset}/{dataset_name}/{speaker_id}')
			continue

		inpath = Path(f'./{input_dataset}/{dataset_name}/{speaker_id}/whisper.json')
		outpath = Path(f'./{output_dataset}/{dataset_name}/{speaker_id}/whisper.json')

		if not inpath.exists():
			continue

		if outpath.exists():
			continue

		os.makedirs(f'./{output_dataset}/{dataset_name}/{speaker_id}/', exist_ok=True)

		try:
			in_metadata = json.loads(open(inpath, 'r', encoding='utf-8').read())
		except Exception as e:
			print("Failed to open metadata file:", inpath)
			continue

		out_metadata = {}
		speaker_metadatas = {}

		for filename, result in in_metadata.items():
			language = result["language"] if "language" in result else "en"
			out_metadata[filename] = {
				"segments": [],
				"language": language,
				"text": "",
				"start": 0,
				"end": 0,
			}
			segments = []
			text = []
			start = 0
			end = 0
			diarized = False

			for segment in result["segments"]:
				# diarize split
				if "speaker" in segment:
					diarized = True
					speaker_id = segment["speaker"]
					if speaker_id not in speaker_metadatas:
						speaker_metadatas[speaker_id] = {}

					if filename not in speaker_metadatas[speaker_id]:
						speaker_metadatas[speaker_id][filename] = {
							"segments": [],
							"language": language,
							"text": "",
							"start": 0,
							"end": 0,
						}

					speaker_metadatas[speaker_id][filename]["segments"].append( segment )
				else:
					segments.append( segment )
				
				text.append( segment["text"] )
				start = min( start, segment["start"] )
				end = max( end, segment["end"] )

			out_metadata[filename]["segments"] = segments
			out_metadata[filename]["text"] = " ".join(text).strip()
			out_metadata[filename]["start"] = start
			out_metadata[filename]["end"] = end

			if len(segments) == 0:
				del out_metadata[filename]

		open(outpath, 'w', encoding='utf-8').write(json.dumps(out_metadata))

		for speaker_id, out_metadata in speaker_metadatas.items():
			os.makedirs(f'./{output_dataset}/{dataset_name}/{speaker_id}/', exist_ok=True)
			outpath = Path(f'./{output_dataset}/{dataset_name}/{speaker_id}/whisper.json')

			open(outpath, 'w', encoding='utf-8').write(json.dumps(out_metadata))