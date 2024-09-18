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

	trim_duration=0,
	min_duration=0,
	max_duration=0,
	
	storage_backend="local"
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

	mfcc = None

	slop = False # should probably have a better name for this, but it governs whether to just sum the entire sequence of embeddings into one embedding to make life easier
	if storage_backend == "faiss":
		slop = True
	elif storage_backend == "chunkdot":
		slop = True
	elif storage_backend == "slop":
		slop = True

	# compute features (embeddings if quantized already, MFCC features if raw audio)
	for filename in tqdm(os.listdir(f'./{speaker_path}/'), desc=f"Encoding '{speaker_path}'", disable=not verbose):
		extension = filename.split(".")[-1]
		filename = filename.replace(f".{extension}", "")

		if text:
			if extension not in artifact_extension:
				raise Exception("!")

			artifact = np.load(f'./{speaker_path}/{filename}.{extension}', allow_pickle=True)[()]
			duration = artifact["metadata"]["original_length"] / artifact["metadata"]["sample_rate"]

			if 0 < min_duration and duration < min_duration:
				continue
			
			if 0 < max_duration and max_duration < duration:
				continue

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
				duration = artifact["metadata"]["original_length"] / artifact["metadata"]["sample_rate"]

				if 0 < min_duration and duration < min_duration:
					continue
				
				if 0 < max_duration and max_duration < duration:
					continue

				qnt = torch.from_numpy(artifact["codes"].astype(int))[0].t().to(dtype=torch.int16, device=device)

				if trim_duration > 0:
					qnt = trim( qnt, int( cfg.dataset.frames_per_second * trim_duration ) )
				
				embedding = tts.audio_embedding( qnt )
			# try and extract features from the raw audio itself
			else:
				# qnt = tts.encode_audio(f'./{speaker_path}/{filename}', trim_length=3.0).to(device)
				wav, sr = load_audio( f'./{speaker_path}/{filename}.{extension}' )

				duration = wav.shape[-1] / sr

				if 0 < min_duration and duration < min_duration:
					continue
				
				if 0 < max_duration and max_duration < duration:
					continue

				if mfcc is None:
					mfcc = T.MFCC(sample_rate=cfg.sample_rate)

				embedding = mfcc(wav.to(device))[0].t()
		
		if slop:
			embedding = embedding.sum(dim=0)

		features[filename] = embedding

	# rely on FAISS to handle storing embeddings and handling queries
	# will probably explode in size fast...........
	if storage_backend == "faiss":
		import faiss
		
		index = faiss.IndexFlatL2( embeddings.shape[-1] )
		embeddings = torch.stack( list( features.values() ) ).cpu()
		index.add( embeddings )

		"""
		# to-do: support just querying for list of similar to cram into JSON metadata
		if verbose:
			for filename, embedding in features.items():
				D, I = index.search(embedding.unsqueeze(0).cpu(), k=2)
				sim = list(I[0][1:])
				print(f'{filename}: {sim}')
		"""

		if metadata_path is not None:
			faiss.write_index(index, str(metadata_path.with_suffix(".faiss")))
		
		return

	"""
	# to-do: actually refine this, maybe
	# desu it's not super easy to install with python3.12, and it is slower than FAISS in testing............
	if storage_backend == "chunkdot":
		from chunkdot import cosine_similarity_top_k

		embeddings = torch.stack( list( features.values() ) ).cpu().numpy()
		similarities = cosine_similarity_top_k(embeddings, top_k=8, show_progress=verbose)

		print(similarities)
		return
	"""

	metadata = None	
	if metadata_path is not None:
		metadata = json.loads(open( metadata_path, "r", encoding="utf-8" ).read()) if metadata_path.exists() else None

	keys = list(features.keys())

	# do batch cosine similarity processing
	if slop:
		embeddings = torch.stack( list( features.values() ) )
		sorted_similarities = {}

		for index, filename in enumerate(keys):
			embedding = features[filename].unsqueeze(0)
			similarities = torch.nn.functional.cosine_similarity(embedding, embeddings, dim=1).cpu().tolist()
			similarities = sorted([ ( keys[i], similarity ) for i, similarity in enumerate( similarities ) if index != i ], key=lambda x: x[1], reverse=True)
			
			sorted_similarities[filename] = similarities

			most_filename, most_score = similarities[0]
			least_filename, least_score = similarities[-1]

			if metadata is not None:
				if filename not in metadata:
					metadata[filename] = {}
				metadata[filename]["similar"] = similarities

			if verbose:
				print( f'{filename}:\n\tMost: {most_filename} ({most_score:.3f})\n\tLeast: {least_filename} ({least_score:.3f})' )

		if metadata is not None:
			with open(str(metadata_path), "w", encoding="utf-8") as f:
				f.write( truncate_json( json.dumps( metadata ) ) )

		return sorted_similarities

	# an EXTREMELY naive implementation, fucking disgusting
	queue = list(combinations(range(len(keys)), 2))
	for key in tqdm(queue, desc="Computing similarities", disable=not verbose):
		index_a, index_b = key
		filename_a, filename_b = keys[index_a], keys[index_b]
		swapped_key = (index_b, index_a)

		if swapped_key in similarities:
			similarities[key] = similarities[swapped_key]
			continue

		if slop:
			embedding_a = features[filename_a]
			embedding_b = features[filename_b]
			
			similarity = torch.nn.functional.cosine_similarity(embedding_a, embedding_b, dim=0).mean().item()
		else:
			shortest = min( features[filename_a].shape[0], features[filename_b].shape[0] )
			embedding_a = features[filename_a][:shortest, :]
			embedding_b = features[filename_b][:shortest, :]

			similarity = torch.nn.functional.cosine_similarity(embedding_a, embedding_b, dim=1).mean().item()
		
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
			f.write( truncate_json( json.dumps( metadata ) ) )

	return sorted_similarities

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--input-speaker", type=Path, default=None)
	parser.add_argument("--use-dataset", action="store_true")

	parser.add_argument("--yaml", type=Path)
	parser.add_argument("--text", action="store_true")
	parser.add_argument("--trim-duration", type=float, default=3.0)
	parser.add_argument("--min-duration", type=float, default=0)
	parser.add_argument("--max-duration", type=float, default=0)
	parser.add_argument("--storage-backend", type=str, default="slop")

	parser.add_argument("--audio-backend", type=str, default="encodec")
	parser.add_argument("--dtype", type=str, default="float32")
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
				metadata_path=cfg.metadata_dir / f'{speaker_name}.faiss',
				yaml=args.yaml,
				text=args.text,
				trim_duration=args.trim_duration,
				min_duration=args.min_duration,
				max_duration=args.max_duration,
				storage_backend=args.storage_backend,

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
			trim_duration=args.trim_duration,
			min_duration=args.min_duration,
			max_duration=args.max_duration,

			audio_backend=args.audio_backend,
			device=args.device,
			dtype=args.dtype,
			amp=args.amp,

			storage_backend=args.storage_backend,
			verbose=True,
		)
	else:
		raise Exception("!")

if __name__ == "__main__":
	main()