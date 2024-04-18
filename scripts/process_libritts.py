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
				# os.rename(f'./{input_dataset}/{dataset_name}/{speaker_id}/{book_id}/{filename}', f'./{output_dataset}/{speaker_id}/{filename}')

				inpath = Path(f'./{input_dataset}/{dataset_name}/{speaker_id}/{book_id}/{filename}')
				outpath = Path(f'./{output_dataset}/{speaker_id}/{filename}')
				
				if ".original.txt" in filename and not _replace_file_extension(outpath, ".json").exists():
					txts.append([inpath, outpath])
				if ".wav" in filename and not _replace_file_extension(outpath, ".dac").exists():
					wavs.append([inpath, outpath])

for paths in tqdm(txts, desc="Phonemizing..."):
	text = open(paths[0], "r", encoding="utf-8").read()
	phones = valle_phonemize(text)
	data = {
		"text": text,
		"phonemes": phones,
		"language": "english",
	}
	open(_replace_file_extension(paths[1], ".json"), 'w', encoding='utf-8').write(json.dumps(data))
	#phones = valle_phonemize(open(paths[0], "r", encoding="utf-8").read())
	#open(_replace_file_extension(paths[1], ".phn.txt"), "w", encoding="utf-8").write(" ".join(phones))

for paths in tqdm(wavs, desc="Quantizing..."):
	qnt = valle_quantize(paths[0], device=device)
	qnt.save(_replace_file_extension(paths[1], ".dac"))
	#torch.save(qnt.cpu(), _replace_file_extension(paths[1], ".qnt.pt"))
