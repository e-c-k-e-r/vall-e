import os
import json
import torch
import torchaudio

from tqdm.auto import tqdm
from pathlib import Path
from vall_e.config import cfg
from vall_e.emb.g2p import encode as valle_phonemize
from vall_e.emb.qnt import encode as valle_quantize, _replace_file_extension

# things that could be args
cfg.sample_rate = 24_000
cfg.inference.audio_backend = "encodec"

input_audio = "voices"
input_metadata = "./training/metadata"
output_dataset = f"./training/data-{'2' if cfg.sample_rate else '4'}4KHz-{cfg.inference.audio_backend}"
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
	
	if dataset_name in ["LibriVox", "Audiobooks"]:
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
					qnt.save(_replace_file_extension(outpath, audio_extension))
				else:
					torch.save( qnt, _replace_file_extension(outpath, audio_extension) )

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
			language = metadata[filename]["language"] if "language" in metadata[filename] else "english"

			if len(metadata[filename]["segments"]) == 0 or not use_slices:
				outpath = Path(f'./{output_dataset}/{dataset_name}/{speaker_id}/{fname}.{extension}')
				text = metadata[filename]["text"]

				if len(text) == 0:
					continue

				if _replace_file_extension(outpath, ".json").exists() and _replace_file_extension(outpath, audio_extension).exists():
					continue

				if not _replace_file_extension(outpath, ".json").exists():
					txts.append((
						outpath,
						text,
						language,
					))
				
				if not _replace_file_extension(outpath, audio_extension).exists():
					if waveform is None:
						waveform, sample_rate = torchaudio.load(inpath)

					wavs.append((
						outpath,
						waveform,
						sample_rate
					))
			else:
				i = 0
				for segment in metadata[filename]["segments"]:
					id = pad(i, 4)
					i = i + 1
					outpath = Path(f'./{output_dataset}/{dataset_name}/{speaker_id}/{fname}_{id}.{extension}')

					if _replace_file_extension(outpath, ".json").exists() and _replace_file_extension(outpath, audio_extension).exists():
						continue

					if not _replace_file_extension(outpath, ".json").exists():
						txts.append((
							outpath,
							segment["text"],
							language,
						))
					
					if not _replace_file_extension(outpath, audio_extension).exists():
						if waveform is None:
							waveform, sample_rate = torchaudio.load(inpath)

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
							waveform[:, start:end],
							sample_rate
						))

		if len(txts) > 0:
			for job in tqdm(txts, desc=f"Phonemizing: {speaker_id}", disable=True):
				outpath, text, language = job
				phones = valle_phonemize(text)
				data = {
					"text": text.strip(),
					"phonemes": phones,
					"language": language,
				}
				open(_replace_file_extension(outpath, ".json"), 'w', encoding='utf-8').write(json.dumps(data))

		if len(wavs) > 0:
			for job in tqdm(wavs, desc=f"Quantizing: {speaker_id}"):
				try:
					outpath, waveform, sample_rate = job
					qnt = valle_quantize(waveform, sr=sample_rate, device=device)
					if cfg.inference.audio_backend == "dac":
						qnt.save(_replace_file_extension(outpath, audio_extension))
					else:
						torch.save( qnt, _replace_file_extension(outpath, audio_extension) )
				except Exception as e:
					print(f"Failed to quantize: {outpath}:", e)
					continue

open("./missing.json", 'w', encoding='utf-8').write(json.dumps(missing))
open("./dataset_list.json", 'w', encoding='utf-8').write(json.dumps(dataset))
