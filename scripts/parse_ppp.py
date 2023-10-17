import os
import json
import torch

from tqdm.auto import tqdm
from pathlib import Path
from vall_e.emb.g2p import encode as valle_phonemize
from vall_e.emb.qnt import encode_from_file as valle_quantize, _replace_file_extension

device = "cuda"

target = "in"

audio_map = {}
text_map = {}

data = {}

for season in os.listdir(f"./{target}/"):
	if not os.path.isdir(f"./{target}/{season}/"):
		continue

	for episode in os.listdir(f"./{target}/{season}/"):
		if not os.path.isdir(f"./{target}/{season}/{episode}/"):
			continue

		for filename in os.listdir(f"./{target}/{season}/{episode}/"):
			path = f'./{target}/{season}/{episode}/{filename}'
			attrs = filename.split("_")
			timestamp = f'{attrs[0]}h{attrs[1]}m{attrs[2]}s'

			key = f'{episode}_{timestamp}'
			
			if filename[-5:] == ".flac":
				name = attrs[3]
				emotion = attrs[4]
				quality = attrs[5]
				
				audio_map[key] = {
					"path": path,
					'episode': episode,
					"name": name,
					"emotion": emotion,
					"quality": quality,
					"timestamp": timestamp,
				}
			
			elif filename[-4:] == ".txt":
				text_map[key] = open(path, encoding="utf-8").read()
txts = {}
wavs = []

for key, entry in audio_map.items():
	path = entry['path']
	name = entry['name']
	emotion = entry['emotion']
	quality = entry['quality']
	episode = entry['episode']
	path = entry['path']
	timestamp = entry['timestamp']
	transcription = text_map[key]
	if name not in data:
		data[name] = {}
		os.makedirs(f'./training/{name}/', exist_ok=True)
		os.makedirs(f'./voices/{name}/', exist_ok=True)

	key = f'{episode}_{timestamp}.flac'
	os.rename(path, f'./voices/{name}/{key}')

	data[name][key] = {
		"segments": [],
		"language": "en",
		"text": transcription,
		"misc": {
			"emotion": emotion,
			"quality": quality,
			"timestamp": timestamp,
			"episode": episode,
		}
	}

	path = f'./voices/{name}/{key}'
	txts[path] = transcription
	wavs.append(Path(path))

for name in data.keys():
	open(f"./training/{name}/whisper.json", "w", encoding="utf-8").write( json.dumps( data[name], indent='\t' ) )

for key, text in tqdm(txts.items(), desc="Phonemizing..."):
	path = Path(key)
	phones = valle_phonemize(text)
	open(_replace_file_extension(path, ".phn.txt"), "w", encoding="utf-8").write(" ".join(phones))

for path in tqdm(wavs, desc="Quantizing..."):
	qnt = valle_quantize(path, device=device)
	torch.save(qnt.cpu(), _replace_file_extension(path, ".qnt.pt"))