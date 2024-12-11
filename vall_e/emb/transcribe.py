"""
# Handles transcribing audio provided through --input-audio
"""

import os
import json
import argparse

import torch
import torchaudio

import whisperx

from functools import cache
from tqdm.auto import tqdm
from pathlib import Path

from ..utils import coerce_dtype


def pad(num, zeroes):
	return str(num).zfill(zeroes+1)

def process_items( items, stride=0, stride_offset=0 ):
	items = sorted( items )
	return items if stride == 0 else [ item for i, item in enumerate( items ) if (i+stride_offset) % stride == 0 ]

# major cringe but should automatically unload models when loading a different one
_cached_models = {
	"model": (None, None),
	"diarization": (None, None),
	"align": (None, None),
}
# yes I can write a decorator to do this
def _load_model(model_name="large-v3", device="cuda", dtype="float16", language="auto"):
	cache_key = f'{model_name}:{device}:{dtype}:{language}'
	if _cached_models["model"][0] == cache_key:
		return _cached_models["model"][1]

	del _cached_models["model"]

	if not isinstance( dtype, str ):
		if dtype == torch.float32:
			dtype = "float32"
		elif dtype == torch.float16:
			dtype = "float16"
		elif dtype == torch.bfloat16:
			dtype = "bfloat16"

	# doesnt support it for some reason
	if dtype == "bfloat16":
		dtype = "float16"
	
	kwargs = {} 
	kwargs["compute_type"] = dtype
	kwargs["task"] = "transcribe"
	kwargs["device"] = device

	if language != "auto":
		kwargs["language"] = language

	model = whisperx.load_model(model_name, **kwargs)

	_cached_models["model"] = (cache_key, model)
	return model

def _load_diarization_model(device="cuda"):
	cache_key = f'{device}'

	if _cached_models["diarization"][0] == cache_key:
		return _cached_models["diarization"][1]
	del _cached_models["diarization"]
	model = whisperx.DiarizationPipeline(device=device)
	_cached_models["diarization"] = (cache_key, model)
	return model

def _load_align_model(language, device="cuda"):
	cache_key = f'{language}:{device}'

	if _cached_models["align"][0] == cache_key:
		return _cached_models["align"][1]
	del _cached_models["align"]
	model = whisperx.load_align_model(language_code=language, device=device)
	_cached_models["align"] = (cache_key, model)
	return model

# yes I can just do a for-loop
def unload_model():
	del _cached_models["model"]
	del _cached_models["diarization"]
	del _cached_models["align"]

	_cached_models["model"] = (None, None)
	_cached_models["diarization"] = (None, None)
	_cached_models["align"] = (None, None)

def transcribe(
	audio,
	language = "auto",
	diarize = False,
	batch_size = 16,
	verbose=False,
	align=True,
	**model_kwargs,
):
	metadata = {
		"segments": [],
		"language": "",
		"text": "",
		"start": 0,
		"end": 0,
	}

	# load requested models
	device = model_kwargs.get("device", "cuda")
	model = _load_model(language=language, **model_kwargs)
	diarize_model = _load_diarization_model(device=device) if diarize else None

	# audio is a path, load it
	if isinstance(audio, str) or isinstance(audio, Path):
		#audio = load_audio(audio)
		audio = whisperx.load_audio(audio)

	result = model.transcribe(audio, batch_size=batch_size)

	if language == "auto":
		language = result["language"]

	if align:
		align_model, align_model_metadata = _load_align_model(language=language, device=device)
		result = whisperx.align(result["segments"], align_model, align_model_metadata, audio, device, return_char_alignments=False)

	if diarize_model is not None:
		diarize_segments = diarize_model(audio)
		result = whisperx.assign_word_speakers(diarize_segments, result)

	text = []
	start = 0
	end = 0
	for segment in result["segments"]:
		text.append( segment["text"] )
		start = min( start, segment["start"] )
		end = max( end, segment["end"] )

	metadata["language"] = language
	metadata["segments"] = result["segments"]
	metadata["text"] = " ".join(text).strip()
	metadata["start"] = start
	metadata["end"] = end

	return metadata

def transcribe_batch(
	input_audio = "voices",
	input_voice = None,
	output_metadata = "training/metadata",
	model_name = "large-v3",
	
	skip_existing = True,
	diarize = False,

	stride = 0,
	stride_offset = 0,

	batch_size = 16,
	device = "cuda",
	dtype = "float16",
):
	# to-do: make this also prepared from args
	language_map = {} # k = group, v = language

	ignore_groups = [] # skip these groups
	ignore_speakers = [] # skip these speakers

	only_groups = [] # only process these groups
	only_speakers = [] # only process these speakers

	if input_voice is not None:
		only_speakers = [input_voice]

	"""
	align_model, align_model_metadata, align_model_language = (None, None, None)
	model =_load_model(model_name, device, compute_type=dtype)
	diarize_model = _load_diarization_model(device=device) if diarize else None
	"""

	for dataset_name in os.listdir(f'./{input_audio}/'):
		if not os.path.isdir(f'./{input_audio}/{dataset_name}/'):
			continue

		if group_name in ignore_groups:
			continue
		if only_groups and group_name not in only_groups:
			continue

		for speaker_id in tqdm(process_items(os.listdir(f'./{input_audio}/{dataset_name}/')), desc="Processing speaker"):
			if not os.path.isdir(f'./{input_audio}/{dataset_name}/{speaker_id}'):
				continue

			if speaker_id in ignore_speakers:
				continue
			if only_speakers and speaker_id not in only_speakers:
				continue

			outpath = Path(f'./{output_metadata}/{dataset_name}/{speaker_id}/whisper.json')

			if outpath.exists():
				metadata = json.loads(open(outpath, 'r', encoding='utf-8').read())
			else:
				os.makedirs(f'./{output_metadata}/{dataset_name}/{speaker_id}/', exist_ok=True)
				metadata = {}

			for filename in tqdm(os.listdir(f'./{input_audio}/{dataset_name}/{speaker_id}/'), desc=f"Processing speaker: {speaker_id}"):
				if skip_existing and filename in metadata:
					continue

				if ".json" in filename:
					continue

				inpath = f'./{input_audio}/{dataset_name}/{speaker_id}/{filename}'

				if os.path.isdir(inpath):
					continue

				metadata[filename] = transcribe( inpath, model_name=model_name, diarize=diarize, device=device, dtype=dtype )
				
				"""
				metadata[filename] = {
					"segments": [],
					"language": "",
					"text": "",
					"start": 0,
					"end": 0,
				}

				audio = whisperx.load_audio(inpath)
				result = model.transcribe(audio, batch_size=batch_size)
				language = result["language"]

				if align_model_language != language:
					tqdm.write(f'Loading language: {language}')
					align_model_language = language
					align_model, align_model_metadata = _load_align_model(language=language, device=device)

				result = whisperx.align(result["segments"], align_model, align_model_metadata, audio, device, return_char_alignments=False)

				metadata[filename]["segments"] = result["segments"]
				metadata[filename]["language"] = language

				if diarize_model is not None:
					diarize_segments = diarize_model(audio)
					result = whisperx.assign_word_speakers(diarize_segments, result)

				text = []
				start = 0
				end = 0
				for segment in result["segments"]:
					text.append( segment["text"] )
					start = min( start, segment["start"] )
					end = max( end, segment["end"] )

				metadata[filename]["text"] = " ".join(text).strip()
				metadata[filename]["start"] = start
				metadata[filename]["end"] = end
				"""

				open(outpath, 'w', encoding='utf-8').write(json.dumps(metadata))

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--input-audio", type=str, default="voices")
	parser.add_argument("--input-voice", type=str, default=None)
	parser.add_argument("--output-metadata", type=str, default="training/metadata")

	parser.add_argument("--model-name", type=str, default="large-v3")
	parser.add_argument("--skip-existing", action="store_true")
	parser.add_argument("--diarize", action="store_true")
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--stride", type=int, default=0)
	parser.add_argument("--stride-offset", type=int, default=0)

	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--dtype", type=str, default="bfloat16")
	parser.add_argument("--amp", action="store_true")
	# parser.add_argument("--raise-exceptions", action="store_true")

	args = parser.parse_args()
	
	# do some assumption magic
	# to-do: find a nice way to spawn multiple processes where tqdm plays nicely
	if args.device.isnumeric():
		args.stride = torch.cuda.device_count()
		args.stride_offset = int(args.device)
		args.device = f'cuda:{args.device}'

	transcribe_batch(
		input_audio = args.input_audio,
		input_voice = args.input_voice,
		output_metadata = args.output_metadata,
		model_name = args.model_name,

		skip_existing = args.skip_existing,
		diarize = args.diarize,

		stride = args.stride,
		stride_offset = args.stride_offset,

		batch_size = args.batch_size,
		device = args.device,
		dtype = args.dtype,
	)

if __name__ == "__main__":
	main()