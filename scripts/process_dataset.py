import os
import json
import torch
import torchaudio
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from vall_e.config import cfg

# things that could be args
cfg.sample_rate = 24_000
cfg.inference.audio_backend = "encodec"
"""
cfg.inference.weight_dtype = "bfloat16"
cfg.inference.dtype = torch.bfloat16
cfg.inference.amp = True
"""

from vall_e.emb.g2p import encode as valle_phonemize
from vall_e.emb.qnt import encode as valle_quantize, _replace_file_extension

input_audio = "voices"
input_metadata = "metadata"
output_dataset = f"training-{'2' if cfg.sample_rate == 24_000 else '4'}4KHz-{cfg.inference.audio_backend}"
device = "cuda"

audio_extension = ".dac" if cfg.inference.audio_backend == "dac" else ".enc"

slice = "auto"
missing = {
	"transcription": [],
	"audio": []
}
dataset = []

def pad(num, zeroes):
	return str(num).zfill(zeroes+1)

for dataset_name in sorted(os.listdir(f'./{input_audio}/')):
	if not os.path.isdir(f'./{input_audio}/{dataset_name}/'):
		print("Is not dir:", f'./{input_audio}/{dataset_name}/')
		continue

	for speaker_id in tqdm(sorted(os.listdir(f'./{input_audio}/{dataset_name}/')), desc=f"Processing speaker in {dataset_name}"):
		if not os.path.isdir(f'./{input_audio}/{dataset_name}/{speaker_id}'):
			print("Is not dir:", f'./{input_audio}/{dataset_name}/{speaker_id}')
			continue
		
		os.makedirs(f'./{output_dataset}/{dataset_name}/{speaker_id}/', exist_ok=True)

		if speaker_id == "Noise":
			for filename in sorted(os.listdir(f'./{input_audio}/{dataset_name}/{speaker_id}/')):
				inpath = Path(f'./{input_audio}/{dataset_name}/{speaker_id}/{filename}')
				outpath = Path(f'./{output_dataset}/{dataset_name}/{speaker_id}/{filename}')

				if _replace_file_extension(outpath, audio_extension).exists():
					continue

				waveform, sample_rate = torchaudio.load(inpath)
				qnt = valle_quantize(waveform, sr=sample_rate, device=device)

				if cfg.inference.audio_backend == "dac":
					np.save(open(_replace_file_extension(outpath, audio_extension), "wb"), {
						"codes": qnt.codes.cpu().numpy().astype(np.uint16),
						"metadata": {
							"original_length": qnt.original_length,
							"sample_rate": qnt.sample_rate,
							
							"input_db": qnt.input_db.cpu().numpy().astype(np.float32),
							"chunk_length": qnt.chunk_length,
							"channels": qnt.channels,
							"padding": qnt.padding,
							"dac_version": "1.0.0",
						},
					})
				else:
					np.save(open(_replace_file_extension(outpath, audio_extension), "wb"), {
						"codes": qnt.cpu().numpy().astype(np.uint16),
						"metadata": {
							"original_length": waveform.shape[-1],
							"sample_rate": sample_rate,
						},
					})

			continue
		
		metadata_path = Path(f'./{input_metadata}/{dataset_name}/{speaker_id}/whisper.json')
		if not metadata_path.exists():
			missing["transcription"].append(str(metadata_path))
			continue

		try:
			metadata = json.loads(open(metadata_path, "r", encoding="utf-8").read())
		except Exception as e:
			missing["transcription"].append(str(metadata_path))
			continue

		if f'{dataset_name}/{speaker_id}' not in dataset:
			dataset.append(f'{dataset_name}/{speaker_id}')

		txts = []
		wavs = []

		use_slices = slice == True or (slice == "auto" and len(metadata.keys()) == 1) or dataset_name in ["LibriVox", "Audiobooks"]

		for filename in sorted(metadata.keys()):
			inpath = Path(f'./{input_audio}/{dataset_name}/{speaker_id}/{filename}')
			if not inpath.exists():
				missing["audio"].append(str(inpath))
				continue
			
			extension = os.path.splitext(filename)[-1][1:]
			fname = filename.replace(f'.{extension}', "")

			waveform, sample_rate = None, None
			language = metadata[filename]["language"] if "language" in metadata[filename] else "en"

			if len(metadata[filename]["segments"]) == 0 or not use_slices:
				outpath = Path(f'./{output_dataset}/{dataset_name}/{speaker_id}/{fname}.{extension}')
				text = metadata[filename]["text"]

				if len(text) == 0:
					continue

				if _replace_file_extension(outpath, audio_extension).exists():
					continue

				if waveform is None:
					waveform, sample_rate = torchaudio.load(inpath)
					if waveform.shape[0] > 1:
						waveform = torch.mean(waveform, dim=0, keepdim=True)

				wavs.append((
					outpath,
					text,
					language,
					waveform,
					sample_rate
				))
			else:
				i = 0
				for segment in metadata[filename]["segments"]:
					id = pad(i, 4)
					i = i + 1

					outpath = Path(f'./{output_dataset}/{dataset_name}/{speaker_id}/{fname}_{id}.{extension}')
					text = segment["text"]

					if len(text) == 0:
						continue

					if _replace_file_extension(outpath, audio_extension).exists():
						continue

					if waveform is None:
						waveform, sample_rate = torchaudio.load(inpath)
						if waveform.shape[0] > 1:
							waveform = torch.mean(waveform, dim=0, keepdim=True)

					start = int(segment['start'] * sample_rate)
					end = int(segment['end'] * sample_rate)

					if start < 0:
						start = 0
					if end >= waveform.shape[-1]:
						end = waveform.shape[-1] - 1

					if end - start < 0:
						continue

					wavs.append((
						outpath,
						text,
						language,
						waveform[:, start:end],
						sample_rate
					))

		if len(wavs) > 0:
			for job in tqdm(wavs, desc=f"Quantizing: {speaker_id}"):
				try:
					outpath, text, language, waveform, sample_rate = job

					phones = valle_phonemize(text)
					qnt = valle_quantize(waveform, sr=sample_rate, device=device)

					if cfg.inference.audio_backend == "dac":
						np.save(open(_replace_file_extension(outpath, audio_extension), "wb"), {
							"codes": qnt.codes.cpu().numpy().astype(np.uint16),
							"metadata": {
								"original_length": qnt.original_length,
								"sample_rate": qnt.sample_rate,
								
								"input_db": qnt.input_db.cpu().numpy().astype(np.float32),
								"chunk_length": qnt.chunk_length,
								"channels": qnt.channels,
								"padding": qnt.padding,
								"dac_version": "1.0.0",

								"text": text.strip(),
								"phonemes": "".join(phones),
								"language": language,
							},
						})
					else:
						np.save(open(_replace_file_extension(outpath, audio_extension), "wb"), {
							"codes": qnt.cpu().numpy().astype(np.uint16),
							"metadata": {
								"original_length": waveform.shape[-1],
								"sample_rate": sample_rate,

								"text": text.strip(),
								"phonemes": "".join(phones),
								"language": language,
							},
						})
				except Exception as e:
					print(f"Failed to quantize: {outpath}:", e)
					continue

open("./missing.json", 'w', encoding='utf-8').write(json.dumps(missing))
open("./dataset_list.json", 'w', encoding='utf-8').write(json.dumps(dataset))