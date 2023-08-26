import os
import json

librilight_dir = "LibriLight-6K"
libritts_dir = "LibriTTS-Train"

librilight_data = {}
libritts_data = {}

for speaker_id in os.listdir(f'./{librilight_dir}/'):
	for filename in os.listdir(f'./{librilight_dir}/{speaker_id}'):
		parts = filename.split("_")
		book_id = parts[1]
		subid = parts[2]

		if speaker_id not in librilight_data:
			librilight_data[speaker_id] = {}
		if book_id not in librilight_data[speaker_id]:
			librilight_data[speaker_id][book_id] = []
		librilight_data[speaker_id][book_id].append(subid)

for speaker_id in os.listdir(f'./{libritts_dir}/'):
	for filename in os.listdir(f'./{libritts_dir}/{speaker_id}'):
		parts = filename.split("_")
		book_id = parts[1]
		subid = parts[2]

		if speaker_id not in libritts_data:
			libritts_data[speaker_id] = {}
		if book_id not in libritts_data[speaker_id]:
			libritts_data[speaker_id][book_id] = []
		libritts_data[speaker_id][book_id].append(subid)

duplicates = []

for speaker_id, books in libritts_data.items():
	if speaker_id not in librilight_data:
		continue
	for book_id, _ in books.items():
		if book_id not in librilight_data[speaker_id]:
			continue
		print(f'Duplicate: {speaker_id}/{book_id}')
		duplicates.append(f'{speaker_id}/{book_id}')

print("Duplicates:", duplicates)