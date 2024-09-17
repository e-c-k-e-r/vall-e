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

from itertools import combinations

_logger = logging.getLogger(__name__)

from tqdm.auto import tqdm
from pathlib import Path

import torchaudio.functional as F
import torchaudio.transforms as T

from ..config import cfg
from ..utils import truncate_json

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

tts = None

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

	maximum_duration=0,
	#use_faiss=True,
):
	global tts

	cfg.set_audio_backend(audio_backend)
	artifact_extension = cfg.audio_backend_extension

	cfg.inference.weight_dtype = dtype # "bfloat16"
	cfg.inference.amp = amp # False

	# easy way to load the model and handle encoding audio
	if tts is None:
		tts = init_tts( yaml=yaml, restart=False, device=device, dtype=dtype )

	queue = []
	features = {}
	similarities = {}
	sorted_similarities = {}

	mfcc = T.MFCC(sample_rate=cfg.sample_rate)

	"""
	# too slow
	if use_faiss:
		import faiss
		index = None
	"""

	# compute features (embeddings if quantized already, MFCC features if raw audio)
	for filename in tqdm(os.listdir(f'./{speaker_path}/'), desc=f"Encoding '{speaker_path}'", disable=not verbose):
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

			embedding = tts.text_embedding( phn )
		else:
			# treat embeddings as features, if provided quantized audio
			if extension in artifact_extension:
				artifact = np.load(f'./{speaker_path}/{filename}.{extension}', allow_pickle=True)[()]
				qnt = torch.from_numpy(artifact["codes"].astype(int))[0].t().to(dtype=torch.int16, device=device)
				if maximum_duration > 0:
					qnt = trim( qnt, int( cfg.dataset.frames_per_second * maximum_duration ) )
				
				embedding = tts.audio_embedding( qnt )
			# try and extract features from the raw audio itself
			else:
				# qnt = tts.encode_audio(f'./{speaker_path}/{filename}', trim_length=3.0).to(device)
				wav, sr = load_audio( f'./{speaker_path}/{filename}.{extension}' )
				embedding = mfcc(wav.to(device))[0].t()
		
		features[filename] = embedding
		
		"""
		if use_faiss:
			if index is None:
				shape = embedding.shape
				index = faiss.IndexFlatL2(shape[1])

			index.add(embedding.cpu())

		if verbose:
			for filename, embedding in features.items():
				D, I = index.search(embedding.cpu(), k=3)
				# print(f'{filename}: {I[1:]}')

		if metadata_path is not None:
			index.save(metadata_path)
		"""

	keys = list(features.keys())
	key_range = range(len(keys))
	# queue = [ (index_a, index_b) for index_b in key_range for index_a in key_range if index_a != index_b ]
	queue = list(combinations(key_range, 2))

	# compute similarities for every utternace
	for key in tqdm(queue, desc="Computing similarities", disable=not verbose):
		index_a, index_b = key
		filename_a, filename_b = keys[index_a], keys[index_b]
		swapped_key = (index_b, index_a)

		if swapped_key in similarities:
			similarities[key] = similarities[swapped_key]
			continue

		shortest = min( features[filename_a].shape[0], features[filename_b].shape[0] )
		similarity = torch.nn.functional.cosine_similarity(features[filename_a][:shortest, :], features[filename_b][:shortest, :], dim=1).mean().item()
		
		similarities[key] = similarity

		# combinations() doesn't have swapped keys
		if swapped_key not in similarities:
			similarities[swapped_key] = similarity

		if index_a not in sorted_similarities:
			sorted_similarities[index_a] = {}
		if index_b not in sorted_similarities[index_a]:
			sorted_similarities[index_a][index_b] = similarity
		
		if index_b not in sorted_similarities:
			sorted_similarities[index_b] = {}	
		if index_a not in sorted_similarities[index_b]:
			sorted_similarities[index_b][index_a] = similarity

	metadata = None	
	if metadata_path is not None:
		if metadata_path.exists():
			metadata = json.loads(open( metadata_path, "r", encoding="utf-8" ).read())
		else:
			metadata = {}

	# sort similarities scores
	for key, sorted_similarity in sorted_similarities.items():
		sorted_similarities[key] = sorted([ ( key, similarity ) for key, similarity in sorted_similarity.items() ], key=lambda x: x[1], reverse=True)

		most_filename, most_score = sorted_similarities[key][0]
		least_filename, least_score = sorted_similarities[key][-1]

		filename = keys[key]

		if metadata is not None:
			if filename not in metadata:
				metadata[filename] = {}
			metadata[filename]["similar"] = sorted_similarities[key]

		#if verbose:
		#	print( f'{filename}:\n\tMost: {most_filename} ({most_score:.3f})\n\tLeast: {least_filename} ({least_score:.3f})' )

	if metadata is not None:
		with open(str(metadata_path), "w", encoding="utf-8") as f:
			serialized = json.dumps( metadata )
			serialized = truncate_json( serialized )
			f.write( serialized )

	return sorted_similarities

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--input-speaker", type=Path, default=None)
	parser.add_argument("--use-dataset", action="store_true")

	parser.add_argument("--yaml", type=Path)
	parser.add_argument("--text", action="store_true")
	parser.add_argument("--maximum-duration", type=float, default=3.0)

	parser.add_argument("--audio-backend", type=str, default="encodec")
	parser.add_argument("--dtype", type=str, default="float16")
	parser.add_argument("--amp", action="store_true")
	parser.add_argument("--device", type=str, default="cpu") # unironically faster
	
	args = parser.parse_args()

	if args.use_dataset:		
		cfg.metadata_dir.mkdir(parents=True, exist_ok=True)

		def add( dir, type="training", audios=True, texts=True ):
			name = str(dir)
			name = name.replace(str(cfg.data_dir), "")
			speaker_name = name
			if "LibriTTS-R" in speaker_name:
				speaker_name = speaker_name.replace("LibriTTS-R", "LibriVox")

			process(
				speaker_path=cfg.data_dir / speaker_name,
				metadata_path=cfg.metadata_dir / f'{speaker_name}.json',
				yaml=args.yaml,
				text=args.text,
				maximum_duration=args.maximum_duration,

				audio_backend=args.audio_backend,
				device=args.device,
				dtype=args.dtype,
				amp=args.amp,

				verbose=True,
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
			maximum_duration=args.maximum_duration,

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