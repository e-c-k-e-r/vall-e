"""
# Handles processing audio provided through --input-audio of adequately annotated transcriptions provided through --input-metadata (through transcribe.py)
# Outputs NumPy objects containing quantized audio and adequate metadata for use of loading in the trainer through --output-dataset
"""

import os
import orjson as json
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
	top_k=8,

	trim_duration=0,
	min_duration=0,
	max_duration=0,
	
	storage_backend="slop"
):
	global tts

	cfg.set_audio_backend(audio_backend)
	artifact_extension = cfg.audio_backend_extension

	cfg.inference.weight_dtype = dtype # "bfloat16"
	cfg.inference.amp = amp # False

	# easy way to load the model and handle encoding audio
	if tts is None:
		tts = init_tts( yaml=yaml, restart=False, device=device, dtype=dtype )

	features = {}

	mfcc = None

	simplified_metadata = True # aims to slim down the raw data in the JSON to store
	slop = True # should probably have a better name for this, but it governs whether to just sum the entire sequence of embeddings into one embedding to make life easier

	# compute features (embeddings if quantized already, MFCC features if raw audio)
	for filename in tqdm(os.listdir(f'./{speaker_path}/'), desc=f"Encoding '{speaker_path.name}'", disable=not verbose):
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

				"""
				if 0 < min_duration and duration < min_duration:
					continue
				
				if 0 < max_duration and max_duration < duration:
					continue
				"""

				qnt = torch.from_numpy(artifact["codes"].astype(int))[0].t().to(dtype=torch.int16, device=device)

				if trim_duration > 0:
					qnt = trim( qnt, int( cfg.dataset.frames_per_second * trim_duration ) )
				
				embedding = tts.audio_embedding( qnt )
			# try and extract features from the raw audio itself
			else:
				# qnt = tts.encode_audio(f'./{speaker_path}/{filename}', trim_length=3.0).to(device)
				wav, sr = load_audio( f'./{speaker_path}/{filename}.{extension}' )

				duration = wav.shape[-1] / sr

				"""
				if 0 < min_duration and duration < min_duration:
					continue
				
				if 0 < max_duration and max_duration < duration:
					continue
				"""

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
				D, I = index.search(embedding.unsqueeze(0).cpu(), k=top_k+1)
				sim = list(I[0][1:])
				print(f'{filename}: {sim}')
		"""
		
		return index
	
	# do batch cosine similarity processing

	keys = list(features.keys())
	embeddings = torch.stack( list( features.values() ) )
	sorted_similarities = {}

	for index, filename in tqdm(enumerate(keys), total=len(keys), desc=f"Computing similarities: {speaker_path.name}"):
		embedding = features[filename].unsqueeze(0)

		similarities = torch.nn.functional.cosine_similarity(embedding, embeddings, dim=1)
		# set current index to -inf
		similarities[index] = float("-inf")
		similarities = torch.topk(similarities, k=top_k, largest=True, sorted=True).indices.tolist()
		# similarities = torch.nn.functional.cosine_similarity(embedding, embeddings, dim=1).cpu().tolist()

		sorted_similarities[filename] = similarities

		# sorting is slow, don't bother
		#sorted_similarities[filename] = sorted([ ( i if simplified_metadata else keys[i], similarity ) for i, similarity in enumerate( similarities ) if index != i ], key=lambda x: x[1], reverse=True)

	return sorted_similarities

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--input-speaker", type=Path, default=None)
	parser.add_argument("--use-dataset", action="store_true")

	parser.add_argument("--yaml", type=Path)
	parser.add_argument("--text", action="store_true")
	# dropped, because this might mess with the indices to map to
	"""
	parser.add_argument("--trim-duration", type=float, default=3.0)
	parser.add_argument("--min-duration", type=float, default=0)
	parser.add_argument("--max-duration", type=float, default=0)
	"""
	parser.add_argument("--storage-backend", type=str, default="slop")
	parser.add_argument("--top-k", type=int, default=8)

	parser.add_argument("--audio-backend", type=str, default="encodec")
	parser.add_argument("--dtype", type=str, default="float16")
	parser.add_argument("--amp", action="store_true")
	parser.add_argument("--device", type=str, default="cuda")
	
	args = parser.parse_args()

	if args.use_dataset:		
		cfg.metadata_dir.mkdir(parents=True, exist_ok=True)

		def add( dir, type="training", audios=True, texts=True ):
			name = str(dir)
			name = name.replace(str(cfg.data_dir), "")
			speaker_name = name
			if "LibriTTS-R" in speaker_name:
				speaker_name = speaker_name.replace("LibriTTS-R", "LibriVox")
			
			metadata_path = cfg.metadata_dir / f'{speaker_name}.json'

			similarities = process(
				speaker_path=cfg.data_dir / speaker_name,
				yaml=args.yaml,
				text=args.text,
				top_k=args.top_k,
				#trim_duration=args.trim_duration,
				#min_duration=args.min_duration,
				#max_duration=args.max_duration,
				storage_backend=args.storage_backend,

				audio_backend=args.audio_backend,
				device=args.device,
				dtype=args.dtype,
				amp=args.amp,

				verbose=True,
			)

			if args.storage_backend == "faiss":
				faiss.write_index(similarities, str(metadata_path.with_suffix(".faiss")))
				return

			metadata = json.loads(open( metadata_path, "r", encoding="utf-8" ).read()) if metadata_path.exists() else {}
			metadata_keys = list(metadata.keys()) if metadata else list(similarities.keys())

			for filename, sim in similarities.items():
				if filename not in metadata:
					metadata[filename] = {}
				
				metadata[filename]["similar"] = sim

			with open(str(metadata_path), "wb") as f:
				f.write( json.dumps( metadata ) )
				#f.write( truncate_json( json.dumps( metadata ) ) )

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
			top_k=args.top_k,
			
			#trim_duration=args.trim_duration,
			#min_duration=args.min_duration,
			#max_duration=args.max_duration,

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