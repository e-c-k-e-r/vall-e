"""
# Handles processing audio provided through --input-audio of adequately annotated transcriptions provided through --input-metadata (through transcribe.py)
# Outputs NumPy objects containing quantized audio and adequate metadata for use of loading in the trainer through --output-dataset
"""

import os
import json
import argparse
import torch
import torchaudio
import numpy as np
import logging

_logger = logging.getLogger(__name__)

from tqdm.auto import tqdm
from pathlib import Path

import torchaudio.functional as F
import torchaudio.transforms as T

from ..config import cfg

# need to validate if this is safe to import before modifying the config
from .g2p import encode as phonemize
from .qnt import encode as quantize, trim, convert_audio

from ..webui import init_tts

def load_audio( path ):
	waveform, sr = torchaudio.load( path )
	# mix channels
	if waveform.shape[0] > 1:
		waveform = torch.mean(waveform, dim=0, keepdim=True)
	# resample
	waveform, sr = convert_audio(waveform, sr, cfg.sample_rate, 1), cfg.sample_rate

	return waveform, sr

def process(
	speaker_path,
	yaml,
	text=False,

	audio_backend="encodec",
	device="cuda",
	dtype="float16",
	amp=False,

	verbose=False,
	metadata_path=None,
):
	cfg.set_audio_backend(audio_backend)
	artifact_extension = cfg.audio_backend_extension

	cfg.inference.weight_dtype = dtype # "bfloat16"
	cfg.inference.amp = amp # False

	# easy way to load the model and handle encoding audio
	tts = init_tts( yaml=yaml, restart=False, device=device, dtype=dtype )

	queue = []
	features = {}
	similarities = {}
	sorted_similarities = {}

	mfcc = T.MFCC(sample_rate=cfg.sample_rate)

	# compute features (embeddings if quantized already, MFCC features if raw audio)
	for filename in tqdm(os.listdir(f'./{speaker_path}/'), desc="Encoding...", disable=not verbose):
		extension = filename.split(".")[-1]
		filename = filename.replace(f".{extension}", "")

		if text:
			if extension not in artifact_extension:
				raise Exception("!")

			artifact = np.load(f'./{speaker_path}/{filename}.{extension}', allow_pickle=True)[()]

			lang = artifact["metadata"]["language"] if "language" in artifact["metadata"]["language"] else "en"
			if "phonemes" in artifact["metadata"]:
				phn = artifact["metadata"]["phonemes"]
			elif "text" in artifact["metadata"]:
				txt = artifact["metadata"]["text"]
				phn = phonemize( txt, language=lang )
			
			phn = phn.replace("(en)", "")
			if lang != "en":
				phn = phn.replace(f"({metadata['language']})", "")

			features[filename] = tts.text_embedding( phn )
		else:
			# treat embeddings as features, if provided quantized audio
			if extension in artifact_extension:
				artifact = np.load(f'./{speaker_path}/{filename}.{extension}', allow_pickle=True)[()]
				qnt = torch.from_numpy(artifact["codes"].astype(int))[0].t().to(dtype=torch.int16, device=device)
				qnt = trim( qnt, int( cfg.dataset.frames_per_second * 3 ) )
				
				features[filename] = tts.audio_embedding( qnt )
			# try and extract features from the raw audio itself
			else:
				# qnt = tts.encode_audio(f'./{speaker_path}/{filename}', trim_length=3.0).to(device)
				wav, sr = load_audio( f'./{speaker_path}/{filename}.{extension}' )
				features[filename] = mfcc(wav.to(device))[0].t()

	# calculate pairs, flattened because it makes tqdm nicer
	for filename_a, embedding_a in features.items():
		for filename_b, embedding_b in features.items():
			if filename_a == filename_b:
				continue

			key = f'{filename_a}:{filename_b}'

			if key in queue:
				continue

			queue.append(key)
		
	# compute similarities for every utternace
	for key in tqdm(queue, desc="Computing similarities", disable=not verbose):
		filename_a, filename_b = key.split(":")
		swapped_key = f'{filename_b}:{filename_a}'
		if swapped_key in similarities:
			similarities[key] = similarities[swapped_key]
			continue

		shortest = min( features[filename_a].shape[0], features[filename_b].shape[0] )
		similarities[key] = torch.nn.functional.cosine_similarity(features[filename_a][:shortest, :], features[filename_b][:shortest, :], dim=1).mean().item()

	# ???
	for key, similarity in similarities.items():
		filename_a, filename_b = key.split(":")

		if filename_a not in sorted_similarities:
			sorted_similarities[filename_a] = {}
		if filename_b not in sorted_similarities[filename_a]:
			sorted_similarities[filename_a][filename_b] = similarity
		
		if filename_b not in sorted_similarities:
			sorted_similarities[filename_b] = {}	
		if filename_a not in sorted_similarities[filename_b]:
			sorted_similarities[filename_b][filename_a] = similarity

	metadata = None	
	if metadata_path is not None and metadata_path.exists():
		metadata = json.loads(open( metadata_path, "r", encoding="utf-8" ).read())

	# sort similarities scores
	for filename, sorted_similarity in sorted_similarities.items():
		sorted_similarities[filename] = sorted([ ( filename, similarity ) for filename, similarity in sorted_similarity.items() ], key=lambda x: x[1], reverse=True)

		most_filename, most_score = sorted_similarities[filename][0]
		least_filename, least_score = sorted_similarities[filename][-1]

		if metadata is not None and filename in metadata:
			metadata[filename] = sorted_similarities[filename]

		if verbose:
			print( f'{filename}:\n\tMost: {most_filename} ({most_score:.3f})\n\tLeast: {least_filename} ({least_score:.3f})' )

	if metadata is not None:
		with open(str(metadata_path), "w", encoding="utf-8") as f:
			f.write( json.dumps( metadata ) )

	return sorted_similarities

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--input-speaker", type=Path, default=None)
	parser.add_argument("--use-dataset", action="store_true")

	parser.add_argument("--yaml", type=Path)
	parser.add_argument("--text", action="store_true")

	parser.add_argument("--audio-backend", type=str, default="encodec")
	parser.add_argument("--dtype", type=str, default="bfloat16")
	parser.add_argument("--amp", action="store_true")
	parser.add_argument("--device", type=str, default="cuda")
	
	args = parser.parse_args()

	if args.use_dataset:		
		root = str(cfg.data_dir)

		cfg.metadata_dir.mkdir(parents=True, exist_ok=True)

		def add( dir, type="training", audios=True, texts=True ):
			name = str(dir)
			name = name.replace(root, "")
			speaker_name = name

			process(
				speaker_path=cfg.data_dir / speaker_name,
				metadata_path=cfg.metadata_dir / f'{speaker_name}.json',
				yaml=args.yaml,
				text=args.text,

				audio_backend=args.audio_backend,
				device=args.device,
				dtype=args.dtype,
				amp=args.amp,

				verbose=False,
			)

		# training
		for data_dir in tqdm(sorted(cfg.dataset.training), desc="Processing Training"):
			add( data_dir, type="training" )

		# validation
		for data_dir in tqdm(sorted(cfg.dataset.validation), desc='Processing Validation'):
			add( data_dir, type="validation" )

		# noise
		for data_dir in tqdm(sorted(cfg.dataset.noise), desc='Processing Noise'):
			add( data_dir, type="noise", texts=False )
	elif args.input_speaker:
		process(
			speaker_path=args.input_speaker,
			yaml=args.yaml,
			text=args.text,

			audio_backend=args.audio_backend,
			device=args.device,
			dtype=args.dtype,
			amp=args.amp,

			verbose=True,
		)
	else:
		raise Exception("!")

if __name__ == "__main__":
	main()