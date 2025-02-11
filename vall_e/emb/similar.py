"""
# Handles processing audio provided through --input-audio of adequately annotated transcriptions provided through --input-metadata (through transcribe.py)
# Outputs NumPy objects containing quantized audio and adequate metadata for use of loading in the trainer through --output-dataset
"""

import os
import argparse
import torch
import torchaudio
import numpy as np
import logging

from itertools import combinations

_logger = logging.getLogger(__name__)

from tqdm.auto import tqdm
from pathlib import Path
from functools import cache

import torchaudio.functional as F
import torchaudio.transforms as T

from ..config import cfg
from ..data import _load_artifact
from ..utils import truncate_json, coerce_dtype
from ..utils.io import json_read, json_write

from .g2p import encode as phonemize
from .qnt import encode as quantize, trim, convert_audio

from ..models import download_model

from ..webui import init_tts

def load_audio( path, target_sr=None ):
	waveform, sr = torchaudio.load( path )
	# mix channels
	if waveform.shape[0] > 1:
		waveform = torch.mean(waveform, dim=0, keepdim=True)
	if target_sr is None:
		target_sr = cfg.sample_rate
	# resample
	waveform, sr = convert_audio(waveform, sr, target_sr, 1), target_sr

	return waveform, sr

tts = None

# this is for computing SIM-O, but can probably technically be used for scoring similar utterances
@cache
def _load_sim_model(device="cuda", dtype="float16", model_name='microsoft/wavlm-large', finetune=False):
	from ..utils.ext.ecapa_tdnn import ECAPA_TDNN_SMALL
	model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large')

	if finetune:
		finetune_path = Path("./data/models/wavlm_large_finetune.pth")
		if not finetune_path.exists():
			download_model(finetune_path)

		state_dict = torch.load( finetune_path )
		state_dict = state_dict['model']
		del state_dict['loss_calculator.projection.weight']
		model.load_state_dict( state_dict )

	model = model.to(device=device, dtype=coerce_dtype(dtype))
	model = model.eval()

	return model, None

	"""
	logging.getLogger('s3prl').setLevel(logging.DEBUG)
	logging.getLogger('speechbrain').setLevel(logging.DEBUG)
	from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
	model = WavLMForXVector.from_pretrained(model_name)
	finetune_path = Path("./data/models/wavlm_large_finetune.pth")
	if finetune_path.exists():
		state_dict = torch.load( finetune_path )
		model.load_state_dict( state_dict['model'] )
	model = model.to(device=device, dtype=coerce_dtype(dtype))
	model = model.eval()
	
	feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

	return model, feature_extractor
	"""

@torch.no_grad()
def speaker_similarity_embedding(
	audio,
	**model_kwargs,
):
	model_kwargs["finetune"] = True
	device = model_kwargs.get("device", "cuda")
	dtype = model_kwargs.get("dtype", "float16")

	model, feature_extractor = _load_sim_model(**model_kwargs)
	
	if isinstance(audio, str) or isinstance(audio, Path):
		audio = load_audio(audio, 16000)

	audio, sr = audio
	embeddings = model(audio.to(device=device, dtype=coerce_dtype(dtype)))
	"""
	features = feature_extractor(audio, return_tensors="pt", sampling_rate=sr)
	features = features.input_values.squeeze(0).to(dtype=coerce_dtype(dtype), device=device)
	
	output = model(input_values=features)
	embeddings = output.embeddings
	embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
	"""
	return embeddings

def batch_similar_utterances(
	speaker_path,
	yaml,

	device="cuda",
	dtype="float16",
	amp=False,

	verbose=False,
	metadata_path=None,
	top_k=8,
	top_p=0.5,
	metadata_keys=[],

	trim_duration=0,
	min_duration=0,
	max_duration=0,
	
	audio_backend="encodec",
	storage_backend="slop",
	similarity_backend="resp",

	return_features=False,
):
	global tts

	cfg.set_audio_backend(audio_backend)
	artifact_extension = cfg.audio_backend_extension

	cfg.inference.weight_dtype = dtype # "bfloat16"
	cfg.inference.amp = amp # False

	# easy way to load the model and handle encoding audio
	if tts is None:
		tts = init_tts( config=yaml, restart=False, device=device, dtype=dtype )

	features = { key: None for key in metadata_keys }

	mfcc = None

	simplified_metadata = True # aims to slim down the raw data in the JSON to store
	slop = True # should probably have a better name for this, but it governs whether to just sum the entire sequence of embeddings into one embedding to make life easier

	if not speaker_path.exists():
		return

	# to-do: find decent thresholds
	"""
	if similarity_backend != "wavlm":
		top_p = float("-inf")
	"""

	# compute features (embeddings if quantized already, MFCC features if raw audio)
	dim_shape = 1024
	for filename in tqdm(os.listdir(f'./{speaker_path}/'), desc=f"Encoding '{speaker_path.name}'", disable=not verbose):
		extension = filename.split(".")[-1]
		filename = filename.replace(f".{extension}", "")

		if filename not in features:
			continue

		if similarity_backend == "text":
			if extension not in artifact_extension:
				raise Exception("!")

			_, metadata = _load_artifact(f'./{speaker_path}/{filename}.{extension}', return_metadata=True)

			"""
			duration = metadata["original_length"] / metadata["sample_rate"]
			if 0 < min_duration and duration < min_duration:
				continue
			
			if 0 < max_duration and max_duration < duration:
				continue
			"""

			lang = metadata["language"] if "language" in metadata["language"] else "en"
			if "phonemes" in metadata:
				phn = metadata["phonemes"]
			elif "text" in metadata:
				txt = metadata["text"]
				phn = phonemize( txt, language=lang )
			
			phn = phn.replace("(en)", "")
			if lang != "en":
				phn = phn.replace(f"({metadata['language']})", "")

			embedding = tts.text_embedding( phn )
		elif similarity_backend == "resp":
			# treat embeddings as features, if provided quantized audio
			if extension not in artifact_extension:
				continue

			qnt, metadata = _load_artifact(f'./{speaker_path}/{filename}.{extension}', return_metadata=True)

			"""
			duration = metadata["original_length"] / metadata["sample_rate"]

			if 0 < min_duration and duration < min_duration:
				continue
			
			if 0 < max_duration and max_duration < duration:
				continue
			"""

			if trim_duration > 0:
				qnt = trim( qnt, int( cfg.dataset.frames_per_second * trim_duration ) )
			
			qnt = qnt.to(device)

			embedding = tts.audio_embedding( qnt )
		# try and extract features from the raw audio itself
		else:
			# qnt = tts.encode_audio(f'./{speaker_path}/{filename}', trim_length=3.0).to(device)
			if similarity_backend == "wavlm":
				embedding = speaker_similarity_embedding( f'./{speaker_path}/{filename}.{extension}' )
			else:
				wav, sr = load_audio( f'./{speaker_path}/{filename}.{extension}' )

				"""
				duration = wav.shape[-1] / sr
				if 0 < min_duration and duration < min_duration:
					continue
				
				if 0 < max_duration and max_duration < duration:
					continue
				"""

				if mfcc is None:
					mfcc = T.MFCC(sample_rate=cfg.sample_rate)

				embedding = mfcc(wav.to(device))[0].t()
		
		dim_shape = embedding.shape[-1]
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
	top_k = min( top_k, len(keys) )

	if top_k == 0:
		top_k = len(keys)

	if len(keys) == 0:
		return None

	# fill any missing keys with a null embedding to keep the order the same
	null_embedding = torch.zeros( (dim_shape,), device=tts.device, dtype=tts.dtype )
	embeddings = torch.stack( [ feature if feature is not None else null_embedding for feature in features.values()  ] )
	sorted_similarities = {}


	for index, filename in tqdm(enumerate(keys), total=len(keys), desc=f"Computing similarities: {speaker_path.name}", disable=not verbose):
		if features[filename] is None:
			continue

		embedding = features[filename].unsqueeze(0)

		similarities = torch.nn.functional.cosine_similarity(embedding, embeddings, dim=1)
		
		# sorting is slow, don't bother
		#sorted_similarities[filename] = sorted([ ( i if simplified_metadata else keys[i], similarity ) for i, similarity in enumerate( similarities ) if index != i ], key=lambda x: x[1], reverse=True)

		# set current index to -inf
		similarities[index] = float("-inf")

		topk = torch.topk(similarities, k=top_k, largest=True, sorted=True)
		similarities = [ (index, keys[index], score) for index, score in zip( topk.indices.tolist(), topk.values.tolist() ) if score > top_p ]

		sorted_similarities[filename] = similarities

	if return_features:
		return sorted_similarities, features

	return sorted_similarities

"""
# (Attempts to) group speakers based on top-k cosine similarities, by pooling together similar utterances together
# It sort of works, but the WavLM finetuned for speaker similarities leaves some false positives without decent threshold values
"""
def sort_similarities(
	path,
	num_speakers,
	out_path=None,
	threshold=0.8,
	orphan_threshold=0.6,
):
	from sklearn.cluster import KMeans

	folders = [ "1", "2", "3", "4", "5", "6-7", "8", "9", "10", "11", "12", "14", "15" ]
	embeddings = json_read(path / "0" / "embeddings.json")

	for filename, embedding in embeddings.items():
		embeddings[filename] = np.array(embedding)

	embeddings_array = np.stack( list( embeddings.values() ) )
	kmeans = KMeans(n_clusters=num_speakers).fit(embeddings_array)

	"""
	if not out_path:
		out_path = path.parent / "speakers.json"

	orphans = []
	speakers = []

	for filename, similarities in metadata.items():
		target = False
		
		# find any existing buckets
		for i, pool in enumerate(speakers):
			for (idx, name, score) in similarities:
				if score and score < threshold:
					continue
				if name in pool:
					target = i
					break
			
			if target != False:
				break
		# not found, create new bucket
		if target == False:
			pool = [ name for (idx, name, score) in similarities if (not score or score > threshold)  ]
			if filename not in pool:
				pool.append(filename)

			# orphan, check later 
			if len(pool) == 1:
				orphans += pool
			else:
				speakers.append(pool)
			continue

		# insert entries into pool
		if filename not in speakers[target]:
			speakers[target].append(filename)

		for (idx, name, score) in similarities:
			if score and score < threshold:
				continue
			if name not in speakers[target]:
				speakers[target].append(name)

	# shove orphans to best scoring pool
	for filename in orphans:
		target = False
		for (idx, name, score) in metadata[filename]:
			if score and score < orphan_threshold:
				continue
			for i, pool in enumerate(speakers):
				if name in pool:
					target = i
					break
			if target != False:
				continue

		if target == False:
			continue
		
		speakers[target].append(filename)
	"""

	json_write( speakers, out_path )

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--input-speaker", type=Path, default=None)
	parser.add_argument("--input-voice", type=str, default=None)
	parser.add_argument("--use-dataset", action="store_true")

	parser.add_argument("--yaml", type=Path)
	parser.add_argument("--out-path", type=Path, default=None)
	# dropped, because this might mess with the indices to map to
	"""
	parser.add_argument("--trim-duration", type=float, default=3.0)
	parser.add_argument("--min-duration", type=float, default=0)
	parser.add_argument("--max-duration", type=float, default=0)
	"""
	parser.add_argument("--top-k", type=int, default=8)
	parser.add_argument("--top-p", type=float, default=0.5)

	parser.add_argument("--storage-backend", type=str, default="slop")
	parser.add_argument("--similarity-backend", type=str, default="resp")
	parser.add_argument("--audio-backend", type=str, default="encodec")

	parser.add_argument("--dtype", type=str, default="float16")
	parser.add_argument("--amp", action="store_true")
	parser.add_argument("--device", type=str, default="cuda")
	
	args = parser.parse_args()

	args.skip_existing = True # 

	if args.use_dataset:		
		cfg.metadata_dir.mkdir(parents=True, exist_ok=True)

		def add( dir, type="training", audios=True, texts=True ):
			name = str(dir)
			name = name.replace(str(cfg.data_dir), "")
			speaker_name = name
			"""
			if "LibriTTS-R" in speaker_name:
				speaker_name = speaker_name.replace("LibriTTS-R", "LibriVox")
			"""

			if args.input_voice and speaker_name != args.input_voice:
				return
			
			metadata_path = cfg.metadata_dir / f'{speaker_name}.json'
			metadata = json_read( metadata_path, default={} )
			metadata_keys = list(metadata.keys()) if metadata else []

			if args.skip_existing and metadata_keys and "similar" in metadata[metadata_keys[-1]]:
				return

			try:
				similarities = batch_similar_utterances(
					speaker_path=cfg.data_dir / speaker_name,
					yaml=args.yaml,
					top_k=args.top_k,
					top_p=args.top_p,
					#trim_duration=args.trim_duration,
					#min_duration=args.min_duration,
					#max_duration=args.max_duration,
					audio_backend=args.audio_backend,
					storage_backend=args.storage_backend,
					similarity_backend=args.similarity_backend,

					metadata_keys=metadata_keys,

					device=args.device,
					dtype=args.dtype,
					amp=args.amp,

					verbose=True,
				)
			except Exception as e:
				similarities = None
			
			if not similarities:
				return

			if args.storage_backend == "faiss":
				faiss.write_index(similarities, str(metadata_path.with_suffix(".faiss")))
				return
			
			for filename, similar in similarities.items():
				if filename not in metadata:
					metadata[filename] = {}
				
				# overkill but i'm very paranoid about mismatching indices
				metadata[filename]["similar"] = [ metadata_keys.index(s[1]) for s in similar ]

			json_write( metadata, metadata_path )

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
		similarities, features = batch_similar_utterances(
			speaker_path=args.input_speaker,
			yaml=args.yaml,
			top_k=args.top_k,
			top_p=args.top_p,

			#trim_duration=args.trim_duration,
			#min_duration=args.min_duration,
			#max_duration=args.max_duration,

			device=args.device,
			dtype=args.dtype,
			amp=args.amp,

			audio_backend=args.audio_backend,
			storage_backend=args.storage_backend,
			similarity_backend=args.similarity_backend,

			verbose=True,
			return_features=True,
		)

		if args.out_path is not None:
			features_json = {}
			for k, v in features.items():
				features_json[k] = [ x.item() for x in v ]

			json_write( similarities, args.out_path / "similarities.json" )
			json_write( features_json, args.out_path / "embeddings.json" )
		else:
			# and print
			for filename, sim in similarities.items():
				print(f'{filename}: {sim}')
	else:
		raise Exception("!")

if __name__ == "__main__":
	main()