import os
import json
import torch

from tqdm.auto import tqdm
from pathlib import Path
from vall_e.emb.g2p import encode as valle_phonemize
from vall_e.emb.qnt import encode_from_file as valle_quantize, _replace_file_extension

input_dataset = "LibriTTS_R"
output_dataset = "LibriTTS-Train"
device = "cuda"

txts = []
wavs = []

for dataset_name in os.listdir(f'./{input_dataset}/'):
	if not os.path.isdir(f'./{input_dataset}/{dataset_name}/'):
		continue

	for speaker_id in tqdm(os.listdir(f'./{input_dataset}/{dataset_name}/'), desc="Processing speaker"):
		if not os.path.isdir(f'./{input_dataset}/{dataset_name}/{speaker_id}'):
			continue
		
		os.makedirs(f'./{output_dataset}/{speaker_id}/', exist_ok=True)
		for book_id in os.listdir(f'./{input_dataset}/{dataset_name}/{speaker_id}'):
			if not os.path.isdir(f'./{input_dataset}/{dataset_name}/{speaker_id}/{book_id}'):
				continue
			for filename in os.listdir(f'./{input_dataset}/{dataset_name}/{speaker_id}/{book_id}'):
				os.rename(f'./{input_dataset}/{dataset_name}/{speaker_id}/{book_id}/{filename}', f'./{output_dataset}/{speaker_id}/{filename}')

				if ".original.txt" in filename:
					txts.append(Path(f'./{output_dataset}/{speaker_id}/{filename}'))
				if ".wav" in filename:
					wavs.append(Path(f'./{output_dataset}/{speaker_id}/{filename}'))

for path in tqdm(txts, desc="Phonemizing..."):
	phones = valle_phonemize(open(path, "r", encoding="utf-8").read())
	open(_replace_file_extension(path, ".phn.txt"), "w", encoding="utf-8").write(" ".join(phones))

for path in tqdm(wavs, desc="Quantizing..."):
	qnt = valle_quantize(path, device=device)
	torch.save(qnt.cpu(), _replace_file_extension(path, ".qnt.pt"))
