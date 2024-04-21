import os
import json

input_dataset = "duplicate"
output_dataset = "LibriLight-4K"

for speaker_id in os.listdir(f'./{input_dataset}/'):
	if not os.path.isdir(f'./{input_dataset}/{speaker_id}/'):
		continue
	for book_name in os.listdir(f'./{input_dataset}/{speaker_id}/'):

		subid = 0
		for filename in os.listdir(f'./{input_dataset}/{speaker_id}/{book_name}'):
			if filename[-5:] != ".json":
				continue

			basename = filename[:-5]

			json_path = f'./{input_dataset}/{speaker_id}/{book_name}/{basename}.json'
			flac_path = f'./{input_dataset}/{speaker_id}/{book_name}/{basename}.flac'

			j = json.load(open(json_path, 'r', encoding="utf-8"))
			id = j['book_meta']['id']
			
			json_id_path = f'./{output_dataset}/{speaker_id}/{speaker_id}_{id}_{subid}.json'
			flac_id_path = f'./{output_dataset}/{speaker_id}/{speaker_id}_{id}_{subid}.flac'

			os.makedirs(f'./{output_dataset}/{speaker_id}/', exist_ok=True)
			os.rename(json_path, json_id_path)
			os.rename(flac_path, flac_id_path)

			subid += 1
