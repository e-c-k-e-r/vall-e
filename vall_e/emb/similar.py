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
	input_speaker,
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
	for filename in tqdm(os.listdir(f'./{input_speaker}/'), desc="Encoding...", disable=not verbose):
		extension = filename.split(".")[-1]

		if text:
			if extension not in artifact_extension:
				raise Exception("!")

			artifact = np.load(f'./{input_speaker}/{filename}', allow_pickle=True)[()]

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
				artifact = np.load(f'./{input_speaker}/{filename}', allow_pickle=True)[()]
				qnt = torch.from_numpy(artifact["codes"].astype(int))[0].t().to(dtype=torch.int16, device=device)
				qnt = trim( qnt, int( cfg.dataset.frames_per_second * 3 ) )
				
				features[filename] = tts.audio_embedding( qnt )
			# try and extract features from the raw audio itself
			else:
				# qnt = tts.encode_audio(f'./{input_speaker}/{filename}', trim_length=3.0).to(device)
				wav, sr = load_audio( f'./{input_speaker}/{filename}' )
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
	if metadata_path is not None:
		metadata_path = metadata_path / f'{input_speaker}.json'
		if metadata_path.exists():
			metadata = json.loads(open( metadata_path, "r", encoding="utf-8" ).read())

	# sort similarities scores
	for key, sorted_similarity in sorted_similarities.items():
		filename_a, filename_b = key.split(":")
		sorted_similarities[key] = sorted([ ( filename, similarity ) for filename, similarity in sorted_similarity.items() ], key=lambda x: x[1], reverse=True)

		most_filename, most_score = sorted_similarities[key][0]
		least_filename, least_score = sorted_similarities[key][-1]

		if metadata is not None and filename_a in metadata:
			metadata[filename_a] = sorted_similarities

		if verbose:
			print( f'{key}:\n\tMost: {most_filename} ({most_score:.3f})\n\tLeast: {least_filename} ({least_score:.3f})' )

	if metadata is not None:
		with open(str(metadata_path), "w", encoding="utf-8") as f:
			f.write( json.dumps( metadata ) )

	return sorted_similarities

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--input-speaker", type=Path)
	parser.add_argument("--yaml", type=Path)
	parser.add_argument("--text", action="store_true")

	parser.add_argument("--audio-backend", type=str, default="encodec")
	parser.add_argument("--dtype", type=str, default="bfloat16")
	parser.add_argument("--amp", action="store_true")
	parser.add_argument("--device", type=str, default="cuda")
	
	args = parser.parse_args()

	process(
		input_speaker=args.input_speaker,
		yaml=args.yaml,
		text=args.text,

		audio_backend=args.audio_backend,
		device=args.device,
		dtype=args.dtype,
		amp=args.amp,

		verbose=True,
	)

if __name__ == "__main__":
	main()