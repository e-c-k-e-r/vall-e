import os
import json

for f in os.listdir(f'./LibriTTS/'):
	if not os.path.isdir(f'./LibriTTS/{f}/'):
		continue
	for j in os.listdir(f'./LibriTTS/{f}/'):
		if not os.path.isdir(f'./LibriTTS/{f}/{j}'):
			continue
		for z in os.listdir(f'./LibriTTS/{f}/{j}'):
			if not os.path.isdir(f'./LibriTTS/{f}/{j}/{z}'):
				continue
			for i in os.listdir(f'./LibriTTS/{f}/{j}/{z}'):
				if i[-4:] != ".wav":
					continue

				os.makedirs(f'./LibriTTS-Train/{j}/', exist_ok=True)
				os.rename(f'./LibriTTS/{f}/{j}/{z}/{i}', f'./LibriTTS-Train/{j}/{i}')