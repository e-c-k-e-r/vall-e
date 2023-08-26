import os
import json

input_dataset = "LibriTTS_R"
output_dataset = "LibriTTS-Train"

for dataset_name in os.listdir(f'./{input_dataset}/'):
	if not os.path.isdir(f'./{input_dataset}/{dataset_name}/'):
		continue
	for speaker_id in os.listdir(f'./{input_dataset}/{dataset_name}/'):
		if not os.path.isdir(f'./{input_dataset}/{dataset_name}/{speaker_id}'):
			continue
		for book_id in os.listdir(f'./{input_dataset}/{dataset_name}/{speaker_id}'):
			if not os.path.isdir(f'./{input_dataset}/{dataset_name}/{speaker_id}/{book_id}'):
				continue
			for filename in os.listdir(f'./{input_dataset}/{dataset_name}/{speaker_id}/{book_id}'):
				if filename[-4:] != ".wav":
					continue

				os.makedirs(f'./{output_dataset}/{speaker_id}/', exist_ok=True)
				os.rename(f'./{input_dataset}/{dataset_name}/{speaker_id}/{book_id}/{filename}', f'./{output_dataset}/{speaker_id}/{filename}')