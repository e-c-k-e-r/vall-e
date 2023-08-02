import os
import json

for f in os.listdir(f'./data/librispeech_finetuning/1h/'):
	for j in os.listdir(f'./data/librispeech_finetuning/1h/{f}/clean'):
		for z in os.listdir(f'./data/librispeech_finetuning/1h/{f}/clean/{j}'):
			for i in os.listdir(f'./data/librispeech_finetuning/1h/{f}/clean/{j}/{z}'):
				os.rename(f'./data/librispeech_finetuning/1h/{f}/clean/{j}/{z}/{i}', f'./data/librilight-tts/{i}')

for j in os.listdir('./data/librispeech_finetuning/9h/clean'):
	for z in os.listdir(f'./data/librispeech_finetuning/9h/clean/{j}'):
		for i in os.listdir(f'./data/librispeech_finetuning/9h/clean/{j}/{z}'):
			os.rename(f'./data/librispeech_finetuning/9h/clean/{j}/{z}/{i}', f'./data/librilight-tts/{i}')

lst = []
for i in os.listdir('./data/librilight-tts/'):
	try:
		if 'trans' not in i:
			continue
		with open(f'./data/librilight-tts/{i}') as f:
			for row in f:
				z = row.split('-')
				name = z[0]+'-'+z[1]+ '-' + z[2].split(' ')[0]
				text = " ".join(z[2].split(' ')[1:])
				lst.append([name, text])
	except Exception as e:
		pass

for i in lst:
	try:
		with open(f'./data/librilight-tts/{i[0]}.txt', 'x') as file:
			file.write(i[1])
	except:
		with open(f'./data/librilight-tts/{i[0]}.txt', 'w+') as file:
			file.write(i[1])

phoneme_map = {}
phoneme_transcript = {}

with open('./data/librispeech_finetuning/phones/phones_mapping.json', 'r') as f:
	phoneme_map_rev = json.load(f)
	for k, v in phoneme_map_rev.items():
		phoneme_map[f'{v}'] = k

with open('./data/librispeech_finetuning/phones/10h_phones.txt', 'r') as f:
	lines = f.readlines()
	for line in lines:
		split = line.strip().split(" ")
		key = split[0]
		tokens = split[1:]

		phonemes = []
		for token in tokens:
			phoneme = phoneme_map[f'{token}']
			phonemes.append( phoneme )

	phoneme_transcript[key] = " ".join(phonemes)

for filename in sorted(os.listdir('./data/librilight-tts')):
	split = filename.split('.')

	key = split[0]
	extension = split[1] # covers double duty of culling .normalized.txt and .phn.txt

	if extension != 'txt':
		continue

	os.rename(f'./data/librilight-tts/{filename}', f'./data/librilight-tts/{key}.normalized.txt')

	if key in phoneme_transcript:
		with open(f'./data/librilight-tts/{key}.phn.txt', 'w', encoding='utf-8') as f:
			f.write(phoneme_transcript[key])