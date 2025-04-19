import sys
import os

argv = os.environ.get('VALLE_ARGS', None)

if argv:
	sys.argv = sys.argv + argv.split(" ")

import re
import math
import argparse
import random
import tempfile
import functools

import torch
import numpy as np

import torchaudio
import gradio as gr

from pathlib import Path


# agony with HF's ZeroGPU spaces
try:
	import spaces

	USING_SPACES = True
	spaces_zerogpu_decorator = spaces.GPU
except Exception as e:
	USING_SPACES = False
	def spaces_zerogpu_decorator(func):
		return func
# more agony, because gradio will not stay launched if directly called from the package, for who knows why
# this allows me to directly copy this file rather than constantly edit it on the HF space repo
if USING_SPACES:
	from vall_e.inference import TTS, cfg
	from vall_e.train import train
	from vall_e.utils import get_devices, setup_logging, timer
	from vall_e.utils.io import json_read, json_stringify
	from vall_e.emb.qnt import decode_to_wave
	from vall_e.data import get_lang_symmap, get_random_prompt
	from vall_e.models.arch import AVAILABLE_ATTENTIONS
	from vall_e.emb.transcribe import transcribe
else:
	from .inference import TTS, cfg
	from .train import train
	from .utils import get_devices, setup_logging, timer
	from .utils.io import json_read, json_stringify
	from .emb.qnt import decode_to_wave
	from .data import get_lang_symmap, get_random_prompt
	from .models.arch import AVAILABLE_ATTENTIONS
	from .emb.transcribe import transcribe


is_windows = sys.platform.startswith("win")

tts = None

layout = {}
layout["inference_tts"] = {}
layout["inference_stt"] = {}
layout["training"] = {}
layout["dataset"] = {}
layout["settings"] = {}

for k in layout.keys():
	layout[k]["inputs"] = { "progress": None }
	layout[k]["outputs"] = {}
	layout[k]["buttons"] = {}

# there's got to be a better way to go about this
def gradio_wrapper(inputs):
	def decorated(fun):
		@functools.wraps(fun)
		def wrapped_function(*args, **kwargs):
			for i, key in enumerate(inputs):
				kwargs[key] = args[i]
			try:
				return fun(**kwargs)
			except Exception as e:
				raise gr.Error(str(e))
		return wrapped_function
	return decorated

# returns a list of models, assuming the models are placed under ./training/ or ./models/ or ./data/models/
def get_model_paths(paths=[Path("./training/"), Path("./models/"), Path("./data/models/")] ):
	configs = []

	for path in paths:
		if not path.exists():
			continue

		for yaml in path.glob("**/*.yaml"):
			if "/logs/" in str(yaml):
				continue
			configs.append( yaml )
		
		for sft in path.glob("**/*.sft"):
			if "/logs/" in str(sft):
				continue
			configs.append( sft )

	configs = [ str(p) for p in configs ]

	return configs

def get_dtypes():
	return ["float32", "float16", "bfloat16", "float8_e5m2", "float8_e4m3fn", "auto"]

def get_attentions():
	return AVAILABLE_ATTENTIONS + ["auto"]

#@gradio_wrapper(inputs=layout["settings"]["inputs"].keys())
def load_model( config, device, dtype, attention ):
	gr.Info(f"Loading: {config}")
	try:
		init_tts( config=Path(config), restart=True, device=device, dtype=dtype, attention=attention )
	except Exception as e:
		raise gr.Error(e)
	gr.Info(f"Loaded model")

def get_speakers():
	return cfg.dataset.training

def get_languages():
	return list(get_lang_symmap().keys()) + ["auto"]

def get_tasks():
	return ["tts", "sr", "ns", "vc"]

#@gradio_wrapper(inputs=layout["dataset"]["inputs"].keys())
def load_sample( speaker ):
	metadata_path = cfg.metadata_dir / f'{speaker}.json'
	metadata = json_read( metadata_path )
	if not metadata:
		raise gr.Error(f"Metadata not found: {metadata_path}")

	key = random.choice( list(metadata.keys()) )
	path = cfg.data_dir / speaker / f'{key}.enc' # to-do: get proper file extension
	data = json_stringify( metadata[key], pretty=True )
	wav, sr = None, None

	if path.exists():
		artifact = np.load(path, allow_pickle=True)[()]
		codes = torch.from_numpy(artifact["codes"].astype(int))[0].t().to(dtype=torch.int16, device=cfg.device)
		wav, sr = decode_to_wave( codes )
		wav = wav.squeeze(0).cpu().numpy()

	return data, (sr, wav)

def gradio_transcribe_input( audio, text, split_by ):
	if not audio:
		return ( text, split_by )
	return ( transcribe( audio, model_name="openai/whisper-base", align=False )["text"], "lines" )

def init_tts(config=None, lora=None, restart=False, device="cuda", dtype="auto", attention=None):
	global tts

	if tts is not None:
		if not restart:
			return tts
		
		del tts
		tts = None
	
	parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
	parser.add_argument("--yaml", type=Path, default=os.environ.get('VALLE_YAML', None)) # os environ so it can be specified in a HuggingFace Space too
	parser.add_argument("--model", type=Path, default=os.environ.get('VALLE_MODEL', None)) # os environ so it can be specified in a HuggingFace Space too
	parser.add_argument("--lora", type=Path, default=os.environ.get('VALLE_LORA', None)) # os environ so it can be specified in a HuggingFace Space too
	parser.add_argument("--device", type=str, default=device)
	parser.add_argument("--amp", action="store_true")
	parser.add_argument("--dtype", type=str, default=dtype)
	parser.add_argument("--attention", type=str, default=attention)
	args, unknown = parser.parse_known_args()

	if config:
		if config.suffix == ".yaml" and not args.yaml:
			args.yaml = config
		elif config.suffix == ".sft" and not args.model:
			args.model = config

	if lora and not args.lora:
		args.lora = lora

	if args.yaml:
		config = args.yaml
	elif args.model:
		config = args.model

	if args.lora:
		lora = args.lora

	tts = TTS( config=config, lora=args.lora, device=args.device, dtype=args.dtype if args.dtype != "auto" else None, amp=args.amp, attention=args.attention )
	return tts

@spaces_zerogpu_decorator
@gradio_wrapper(inputs=layout["inference_tts"]["inputs"].keys())
def do_inference_tts( progress=gr.Progress(track_tqdm=True), *args, **kwargs ):
	if not cfg.models:
		raise Exception("No model loaded.")

	if kwargs.pop("dynamic-sampling", False):
		kwargs['min-ar-temperature'] = 0.01 if kwargs['ar-temperature'] > 0.01 else 0.0
		kwargs['min-nar-temperature'] = 0.0 # 0.85 if kwargs['nar-temperature'] > 0.85 else 0.0 # should probably disable it for the NAR
	else:
		kwargs['min-ar-temperature'] = -1
		kwargs['min-nar-temperature'] = -1

	parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
	# I'm very sure I can procedurally generate this list
	parser.add_argument("--text", type=str, default=kwargs["text"])
	parser.add_argument("--task", type=str, default=kwargs["task"])
	parser.add_argument("--modality", type=str, default=kwargs["modality"])
	parser.add_argument("--references", type=str, default=kwargs["reference"])
	parser.add_argument("--voice-convert", type=str, default=kwargs["voice-convert"])
	parser.add_argument("--language", type=str, default=kwargs["language"])
	parser.add_argument("--text-language", type=str, default=kwargs["text-language"])
	parser.add_argument("--no-phonemize", action="store_true")
	parser.add_argument("--play", action="store_true")
	parser.add_argument("--split-text-by", type=str, default=kwargs["split-text-by"])
	parser.add_argument("--context-history", type=int, default=kwargs["context-history"])
	parser.add_argument("--input-prompt-length", type=float, default=kwargs["input-prompt-length"])
	#parser.add_argument("--input-prompt-prefix", action='store_true', default=kwargs["input-prompt-prefix"])
	parser.add_argument("--max-duration", type=int, default=int(kwargs["max-duration"]*cfg.dataset.frames_per_second))
	#parser.add_argument("--max-levels", type=int, default=kwargs["max-levels"])
	parser.add_argument("--max-steps", type=int, default=kwargs["max-steps"])
	parser.add_argument("--ar-temperature", type=float, default=kwargs["ar-temperature"])
	parser.add_argument("--nar-temperature", type=float, default=kwargs["nar-temperature"])
	parser.add_argument("--min-ar-temperature", type=float, default=kwargs["min-ar-temperature"])
	parser.add_argument("--min-nar-temperature", type=float, default=kwargs["min-nar-temperature"])
	#parser.add_argument("--prefix-silence", type=float, default=kwargs["prefix-silence"])
	parser.add_argument("--top-p", type=float, default=kwargs["top-p"])
	parser.add_argument("--top-k", type=int, default=kwargs["top-k"])
	parser.add_argument("--top-no", type=float, default=kwargs["top-no"])
	parser.add_argument("--min-p", type=float, default=kwargs["min-p"])
	parser.add_argument("--repetition-penalty", type=float, default=kwargs["repetition-penalty"])
	parser.add_argument("--repetition-penalty-decay", type=float, default=kwargs["repetition-penalty-decay"])
	parser.add_argument("--length-penalty", type=float, default=kwargs["length-penalty"])
	"""
	parser.add_argument("--beam-width", type=int, default=kwargs["beam-width"])
	parser.add_argument("--mirostat-tau", type=float, default=kwargs["mirostat-tau"])
	parser.add_argument("--mirostat-eta", type=float, default=kwargs["mirostat-eta"])
	parser.add_argument("--dry-multiplier", type=float, default=kwargs["dry-multiplier"])
	parser.add_argument("--dry-base", type=float, default=kwargs["dry-base"])
	parser.add_argument("--dry-allowed-length", type=int, default=kwargs["dry-allowed-length"])
	parser.add_argument("--entropix-sampling", action="store_true")
	parser.add_argument("--layer-skip", action="store_true")
	parser.add_argument("--layer-skip-exit-layer", type=int, default=kwargs["layer-skip-exit-layer"])
	parser.add_argument("--layer-skip-entropy-threshold", type=int, default=kwargs["layer-skip-entropy-threshold"])
	parser.add_argument("--layer-skip-varentropy-threshold", type=int, default=kwargs["layer-skip-varentropy-threshold"])
	"""
	parser.add_argument("--refine-on-stop", action="store_true")
	parser.add_argument("--denoise-start", type=float, default=0.0)
	parser.add_argument("--cfg-strength", type=float, default=kwargs['cfg-strength'])
	parser.add_argument("--cfg-rescale", type=float, default=kwargs['cfg-rescale'])
	
	parser.add_argument("--sampling-scores-masked-only", action="store_true")
	parser.add_argument("--sampling-scores-flatten", action="store_true")
	parser.add_argument("--sampling-scores-remask", action="store_true")

	args, unknown = parser.parse_known_args()

	if is_windows:
		tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
	else:
		tmp = tempfile.NamedTemporaryFile(suffix='.wav')

	"""
	if not args.references:
		raise Exception("No reference audio provided.")
	"""

	if kwargs.pop("entropix-sampling", False):
		args.entropix_sampling = True
	
	if kwargs.pop("layer-skip", False):
		args.layer_skip = True
	
	if kwargs.pop("refine-on-stop", False):
		args.refine_on_stop = True

	if kwargs.pop("no-phonemize", False):
		args.no_phonemize = True
	
	if kwargs.pop("play", False):
		args.play = True
	
	if kwargs.pop("sampling-scores-masked-only", False):
		args.sampling_scores_masked_only = True
	
	if kwargs.pop("sampling-scores-flatten", False):
		args.sampling_scores_flatten = True
	
	if kwargs.pop("sampling-scores-remask", False):
		args.sampling_scores_remask = True

	if args.split_text_by == "lines":
		args.split_text_by = "\n"
	elif args.split_text_by == "none":
		args.split_text_by = None

	if args.text_language == "auto":
		args.text_language = None

	tts = init_tts()
	
	gr.Info(f"Inferencing... (Modality: {tts.modality(args.modality.lower())})")

	sampling_kwargs = dict(
		split_text_by=args.split_text_by,
		context_history=args.context_history,
		phonemize=not args.no_phonemize,
		voice_convert=args.voice_convert,
		max_steps=args.max_steps,
		#max_levels=args.max_levels,
		max_duration=args.max_duration,
		ar_temperature=args.ar_temperature, nar_temperature=args.nar_temperature,
		min_ar_temperature=args.min_ar_temperature, min_nar_temperature=args.min_nar_temperature,
		top_p=args.top_p, top_k=args.top_k, min_p=args.min_p, top_no=args.top_no,
		repetition_penalty=args.repetition_penalty, repetition_penalty_decay=args.repetition_penalty_decay,
		length_penalty=args.length_penalty,
		#beam_width=args.beam_width,
		#mirostat_tau=args.mirostat_tau, mirostat_eta=args.mirostat_eta,
		#dry_multiplier=args.dry_multiplier, dry_base=args.dry_base, dry_allowed_length=args.dry_allowed_length,
		#entropix_sampling=args.entropix_sampling,
		#layer_skip=args.layer_skip,
		#layer_skip_exit_layer=args.layer_skip_exit_layer,
		#layer_skip_entropy_threshold=args.layer_skip_entropy_threshold,
		#layer_skip_varentropy_threshold=args.layer_skip_varentropy_threshold,
		#refine_on_stop=args.refine_on_stop,
		denoise_start=args.denoise_start,
		#prefix_silence=args.prefix_silence,
		#input_prompt_prefix=args.input_prompt_prefix,
		input_prompt_length=args.input_prompt_length,
		cfg_strength=args.cfg_strength,
		cfg_rescale=args.cfg_rescale,

		sampling_scores_masked_only=args.sampling_scores_masked_only,
		sampling_scores_flatten=args.sampling_scores_flatten,
		sampling_scores_remask=args.sampling_scores_remask,
	)

	with timer("Inferenced in", callback=lambda msg: gr.Info( msg )) as t:
		wav, sr = tts.inference(
			text=args.text,
			language=args.language,
			text_language=args.text_language,
			task=args.task,
			play=args.play,
			modality=args.modality.lower(),
			references=args.references.split(";") if args.references is not None else [],
			**sampling_kwargs,
		)
	
	wav = wav.squeeze(0).cpu().numpy()
	return (sr, wav)

@gradio_wrapper(inputs=layout["inference_stt"]["inputs"].keys())
def do_inference_stt( progress=gr.Progress(track_tqdm=True), *args, **kwargs ):
	if not cfg.models:
		raise Exception("No model loaded.")

	if kwargs.pop("dynamic-sampling", False):
		kwargs['min-ar-temperature'] = 0.85 if kwargs['ar-temperature'] > 0.85 else 0.0
	else:
		kwargs['min-ar-temperature'] = -1

	parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
	# I'm very sure I can procedurally generate this list
	parser.add_argument("--task", type=str, default="stt")
	parser.add_argument("--references", type=str, default=kwargs["reference"])
	parser.add_argument("--max-duration", type=int, default=0)
	parser.add_argument("--language", type=str, default=kwargs["language"])
	parser.add_argument("--ar-temperature", type=float, default=kwargs["ar-temperature"])
	parser.add_argument("--min-ar-temperature", type=float, default=kwargs["min-ar-temperature"])
	parser.add_argument("--top-p", type=float, default=kwargs["top-p"])
	parser.add_argument("--top-k", type=int, default=kwargs["top-k"])
	parser.add_argument("--min-p", type=float, default=kwargs["min-p"])
	parser.add_argument("--repetition-penalty", type=float, default=kwargs["repetition-penalty"])
	parser.add_argument("--repetition-penalty-decay", type=float, default=kwargs["repetition-penalty-decay"])
	parser.add_argument("--length-penalty", type=float, default=kwargs["length-penalty"])
	parser.add_argument("--beam-width", type=int, default=kwargs["beam-width"])
	parser.add_argument("--mirostat-tau", type=float, default=kwargs["mirostat-tau"])
	parser.add_argument("--mirostat-eta", type=float, default=kwargs["mirostat-eta"])
	parser.add_argument("--dry-multiplier", type=float, default=kwargs["dry-multiplier"])
	parser.add_argument("--dry-base", type=float, default=kwargs["dry-base"])
	parser.add_argument("--dry-allowed-length", type=int, default=kwargs["dry-allowed-length"])
	args, unknown = parser.parse_known_args()

	"""
	if not args.references:
		raise Exception("No reference audio provided.")
	"""

	args.references = args.references.split(";") if args.references is not None else []
	if args.max_duration == 0:
		for i, path in enumerate( args.references ):
			metadata = torchaudio.info(path)
			duration = metadata.num_frames / metadata.sample_rate
			args.max_duration += duration
		args.max_duration = math.floor( args.max_duration * 20 ) # assume 20 tokens per second
	
	if kwargs.pop("entropix-sampling", False):
		args.entropix_sampling = True

	tts = init_tts()

	sampling_kwargs = dict(
		max_duration=args.max_duration,
		ar_temperature=args.ar_temperature,
		min_ar_temperature=args.min_ar_temperature,
		top_p=args.top_p, top_k=args.top_k, min_p=args.min_p,
		repetition_penalty=args.repetition_penalty, repetition_penalty_decay=args.repetition_penalty_decay,
		length_penalty=args.length_penalty,
		beam_width=args.beam_width,
		mirostat_tau=args.mirostat_tau, mirostat_eta=args.mirostat_eta,
		dry_multiplier=args.dry_multiplier, dry_base=args.dry_base, dry_allowed_length=args.dry_allowed_length,
	)
	
	gr.Info("Inferencing...")
	with timer("Inferenced in") as t:
		text = tts.inference(
			text="",
			language=args.language,
			task="stt",
			references=args.references,
			**sampling_kwargs,
		)
	
	return text

"""
@gradio_wrapper(inputs=layout["training"]["inputs"].keys())
def do_training( progress=gr.Progress(track_tqdm=True), *args, **kwargs ):

	while True:
		metrics = next(it)
		yield metrics
"""	

# setup args
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--yaml", type=Path, default=os.environ.get('VALLE_YAML', None)) # os environ so it can be specified in a HuggingFace Space too
parser.add_argument("--model", type=Path, default=os.environ.get('VALLE_MODEL', None)) # os environ so it can be specified in a HuggingFace Space too
parser.add_argument("--listen", default=None, help="Path for Gradio to listen on")
parser.add_argument("--share", action="store_true")
parser.add_argument("--render_markdown", action="store_true", default="VALLE_YAML" in os.environ)
args, unknown = parser.parse_known_args()

args.listen_host = None
args.listen_port = None
args.listen_path = None
if args.listen:
	try:
		match = re.findall(r"^(?:(.+?):(\d+))?(\/.*?)?$", args.listen)[0]

		args.listen_host = match[0] if match[0] != "" else "127.0.0.1"
		args.listen_port = match[1] if match[1] != "" else None
		args.listen_path = match[2] if match[2] != "" else "/"
	except Exception as e:
		pass

if args.listen_port is not None:
	args.listen_port = int(args.listen_port)
	if args.listen_port == 0:
		args.listen_port = None

# setup gradio
ui = gr.Blocks()
with ui:
	with gr.Tab("Inference"):
		with gr.Tab("Text-to-Speech"):
			with gr.Row():
				with gr.Column(scale=8):
					with gr.Tab("Text"):
						layout["inference_tts"]["inputs"]["text"] = gr.Textbox(lines=5, value=get_random_prompt, label="Input Prompt")
					with gr.Tab("Speech"):
						layout["inference_tts"]["inputs"]["voice-convert"] = gr.Audio(label="Audio Input", sources=["upload"], type="filepath") # , info="Guiding utternace.")
			with gr.Row():
				with gr.Column(scale=1):
					layout["inference_tts"]["inputs"]["reference"] = gr.Audio(label="Audio Input", sources=["upload"], type="filepath") # , info="Reference audio for TTS")
					# layout["inference_tts"]["stop"] = gr.Button(value="Stop")
					layout["inference_tts"]["outputs"]["output"] = gr.Audio(label="Output")
					layout["inference_tts"]["buttons"]["inference"] = gr.Button(value="Inference")
				with gr.Column(scale=7):
					with gr.Tab("Basic Settings"):
						with gr.Row():
							layout["inference_tts"]["inputs"]["max-steps"] = gr.Slider(value=50, minimum=1, maximum=200, step=1, label="Max Steps", info="Limits how many steps to perform in the NAR-len (demask) pass.")
							layout["inference_tts"]["inputs"]["max-duration"] = gr.Slider(value=12, minimum=1, maximum=32, step=0.1, label="Maximum Duration", info="Limits how long an utterance can be.")
							layout["inference_tts"]["inputs"]["input-prompt-length"] = gr.Slider(value=0.0, minimum=0.0, maximum=12.0, step=0.5, label="Input Prompt Repeat/Trim Length", info="Repeats/trims the input prompt down to X seconds (0 to disable).")
						with gr.Row():
							layout["inference_tts"]["inputs"]["text-language"] = gr.Dropdown(choices=get_languages(), label="Language (Text)", value="auto", info="Language the input text is in.")
							layout["inference_tts"]["inputs"]["language"] = gr.Dropdown(choices=get_languages(), label="Language (Output)", value="auto", info="Target language/accent to output.")
							layout["inference_tts"]["inputs"]["task"] = gr.Dropdown(choices=get_tasks(), label="Task", value="tts", info="")
						with gr.Row():
							layout["inference_tts"]["inputs"]["split-text-by"] = gr.Dropdown(choices=["sentences", "lines"], label="Text Delimiter", info="How to split the text into utterances.", value="sentences")
							layout["inference_tts"]["inputs"]["context-history"] = gr.Slider(value=0, minimum=0, maximum=4, step=1, label="(Rolling) Context History", info="How many prior lines to serve as the context/prefix (0 to disable).")
						with gr.Row():
							layout["inference_tts"]["inputs"]["no-phonemize"] = gr.Checkbox(label="No Phonemize", info="Use raw text rather than phonemize the text as the input prompt.")
							layout["inference_tts"]["inputs"]["play"] = gr.Checkbox(label="Auto Play", info="Auto play on generation (using sounddevice).")
					with gr.Tab("Sampler Settings"):
						with gr.Row():
							layout["inference_tts"]["inputs"]["ar-temperature"] = gr.Slider(value=1.0, minimum=0.0, maximum=1.5, step=0.05, label="Temperature (AR/NAR-len)", info="Adjusts the probabilities in the AR/NAR-len. (0 to greedy* sample)")
							layout["inference_tts"]["inputs"]["nar-temperature"] = gr.Slider(value=0.0, minimum=0.0, maximum=1.5, step=0.05, label="Temperature (NAR)", info="Adjusts the probabilities in the NAR. (0 to greedy sample)")
							layout["inference_tts"]["inputs"]["modality"] = gr.Dropdown(value="Auto", choices=["Auto", "AR+NAR", "NAR-len"], label="Modality", info="Whether to inference with the AR+NAR or through the NAR-len.")
						with gr.Row():
							layout["inference_tts"]["inputs"]["cfg-strength"] = gr.Slider(value=1.0, minimum=0.0, maximum=14.0, step=0.5, label="CFG Strength", info="Classifier Free Guidance scale (AR needs 1, NAR-len needs 3).")
							layout["inference_tts"]["inputs"]["cfg-rescale"] = gr.Slider(value=0.75, minimum=0.0, maximum=1.0, step=0.05, label="CFG Rescale (Phi)", info="Factor when rescaling for Classifier Free Guidance (0 to disable).")
						with gr.Row():
							layout["inference_tts"]["inputs"]["min-p"] = gr.Slider(value=0.0, minimum=0.0, maximum=1.0, step=0.05, label="Min P", info="Filter out logits lower than this value.")
							layout["inference_tts"]["inputs"]["top-p"] = gr.Slider(value=1.0, minimum=0.0, maximum=1.0, step=0.05, label="Top P", info=r"Limits the samples that are outside the top P% of probabilities.")
							layout["inference_tts"]["inputs"]["top-k"] = gr.Slider(value=0, minimum=0, maximum=1024, step=1, label="Top K", info="Limits the samples to the top K of probabilities.")
							layout["inference_tts"]["inputs"]["top-no"] = gr.Slider(value=0, minimum=0, maximum=2, step=0.5, label="Top-nσ", info="Performs top-nσ logits processing.")
						with gr.Row():
							layout["inference_tts"]["inputs"]["repetition-penalty"] = gr.Slider(value=1.0, minimum=0.0, maximum=5.0, step=0.05, label="Repetition Penalty", info="Incurs a penalty to tokens based on how often they appear in a sequence.")
							layout["inference_tts"]["inputs"]["repetition-penalty-decay"] = gr.Slider(value=0.0, minimum=-2.0, maximum=2.0, step=0.05, label="Repetition Penalty Length Decay", info="Modifies the reptition penalty based on how far back in time the token appeared in the sequence.")
							layout["inference_tts"]["inputs"]["length-penalty"] = gr.Slider(value=0.0, minimum=-2.0, maximum=2.0, step=0.05, label="Length Penalty", info="(AR only) Modifies the probability of a stop token based on the current length of the sequence.")
						with gr.Row():
							layout["inference_tts"]["inputs"]["sampling-scores-masked-only"] = gr.Checkbox(label="Sampled Scores: Masked Only", info="(NAR-len only) Update scores for newly generated tokens only")
							layout["inference_tts"]["inputs"]["sampling-scores-flattened"] = gr.Checkbox(label="Sampled Scores: Flattened", info="(NAR-len only) Flattens the scores for all codebook levels")
							layout["inference_tts"]["inputs"]["sampling-scores-remask"] = gr.Checkbox(label="Sampled Scores: Remask", info="(NAR-len only) Remasks P%% of existing tokens randomly after each step.")
					# These settings are pretty much not supported anyways
					"""
					with gr.Tab("Experimental Settings", visible=cfg.experimental):
						with gr.Row():
							layout["inference_tts"]["inputs"]["max-levels"] = gr.Slider(value=7, minimum=0, maximum=7, step=1, label="Max NAR Levels", info="Limits how many steps to perform in the NAR pass.")
							layout["inference_tts"]["inputs"]["beam-width"] = gr.Slider(value=0, minimum=0, maximum=32, step=1, label="Beam Width", info="Number of branches to search through for beam search sampling.")
							layout["inference_tts"]["inputs"]["prefix-silence"] = gr.Slider(value=0.0, minimum=0.0, maximum=1.0, step=0.5, label="Silence Prefix Duration", info="Amount of silence to prefix to the output response before beginning inference.")
						with gr.Row():
							layout["inference_tts"]["inputs"]["input-prompt-prefix"] = gr.Checkbox(label="Input Prompt as Prefix", info="Treats the input prompt clip as the prefix of the generated sequence.")
							layout["inference_tts"]["inputs"]["dynamic-sampling"] = gr.Checkbox(label="Dynamic Temperature", info="Dynamically adjusts the temperature based on the highest confident predicted token per sampling step.")
							layout["inference_tts"]["inputs"]["entropix-sampling"] = gr.Checkbox(label="Entropix Sampling", info="Dynamically samples based on entropy/varentropy values from the logits / attention scores.")
							layout["inference_tts"]["inputs"]["refine-on-stop"] = gr.Checkbox(label="Refine on <stop>", info="Uses the last step's logits for the AR sequence instead.")
						with gr.Row():
							layout["inference_tts"]["inputs"]["mirostat-tau"] = gr.Slider(value=0.0, minimum=0.0, maximum=8.0, step=0.05, label="Mirostat τ (Tau)", info="The \"surprise\" value when performing mirostat sampling. 0 to disable.")
							layout["inference_tts"]["inputs"]["mirostat-eta"] = gr.Slider(value=0.0, minimum=0.0, maximum=2.0, step=0.05, label="Mirostat η (Eta)", info="The \"learning rate\" during mirostat sampling applied to the maximum surprise.")
						with gr.Row():
							layout["inference_tts"]["inputs"]["dry-multiplier"] = gr.Slider(value=0.0, minimum=0.0, maximum=8.0, step=0.05, label="DRY Multiplier", info="The multiplying factor for the DRY score penalty (0 to disable DRY sampling).")
							layout["inference_tts"]["inputs"]["dry-base"] = gr.Slider(value=1.75, minimum=0.0, maximum=8.0, step=0.05, label="DRY Base", info="The base of the exponent in the DRY score penalty")
							layout["inference_tts"]["inputs"]["dry-allowed-length"] = gr.Slider(value=2, minimum=0, maximum=75, step=1, label="Allowed Length", info="The maximimum length a token can be to perform DRY penalty with.")
						with gr.Row():
							layout["inference_tts"]["inputs"]["layer-skip"] = gr.Checkbox(label="Layer Skip", info="Performs self-speculative early exit 'sampling'")
							layout["inference_tts"]["inputs"]["layer-skip-exit-layer"] = gr.Slider(value=11, minimum=0, maximum=11, step=1, label="Layer Skip Exit Layer", info="Maximum model layer to exit early from.")
							layout["inference_tts"]["inputs"]["layer-skip-entropy-threshold"] = gr.Slider(value=0.1, minimum=0, maximum=1.0, step=0.01, label="Layer Skip Entropy Threshold", info="Entropy threshold for early-exit")
							layout["inference_tts"]["inputs"]["layer-skip-varentropy-threshold"] = gr.Slider(value=0.1, minimum=0, maximum=1.0, step=0.01, label="Layer Skip Varentropy Threshold", info="Varentropy threshold for early-exit")
					"""

		layout["inference_tts"]["buttons"]["inference"].click(
			fn=do_inference_tts,
			inputs=[ x for x in layout["inference_tts"]["inputs"].values() if x is not None],
			outputs=[ x for x in layout["inference_tts"]["outputs"].values() if x is not None]
		)

		# IC
		layout["inference_tts"]["inputs"]["voice-convert"].change(
			gradio_transcribe_input,
			[
				layout["inference_tts"]["inputs"]["voice-convert"],
				layout["inference_tts"]["inputs"]["text"],
				layout["inference_tts"]["inputs"]["split-text-by"],
			],
			[
				layout["inference_tts"]["inputs"]["text"],
				layout["inference_tts"]["inputs"]["split-text-by"],
			]
		)

		with gr.Tab("Speech to Text"):
			with gr.Row():
				with gr.Column(scale=8):
					layout["inference_stt"]["outputs"]["ouput"] = gr.Textbox(lines=1, label="Output Transcription")
			with gr.Row():
				with gr.Column(scale=1):
					layout["inference_stt"]["inputs"]["reference"] = gr.Audio(label="Audio Input", sources=["upload"], type="filepath") #, info="Reference audio for TTS")
					# layout["inference_stt"]["stop"] = gr.Button(value="Stop")
					layout["inference_stt"]["buttons"]["inference"] = gr.Button(value="Inference")
				with gr.Column(scale=7):
					with gr.Tab("Basic Settings"):
						with gr.Row():
							layout["inference_stt"]["inputs"]["ar-temperature"] = gr.Slider(value=0.0, minimum=0.0, maximum=1.5, step=0.05, label="Temperature (AR)", info="Modifies the randomness from the samples in the AR. (0 to greedy sample)")
							layout["inference_stt"]["inputs"]["language"] = gr.Dropdown(choices=get_languages(), label="Language", value="en", info="Language of the input audio being transcribed.")
					with gr.Tab("Sampler Settings", visible=False):
						with gr.Row():
							layout["inference_stt"]["inputs"]["top-p"] = gr.Slider(value=1.0, minimum=0.0, maximum=1.0, step=0.05, label="Top P", info=r"Limits the samples that are outside the top P% of probabilities.")
							layout["inference_stt"]["inputs"]["top-k"] = gr.Slider(value=0, minimum=0, maximum=1024, step=1, label="Top K", info="Limits the samples to the top K of probabilities.")
							layout["inference_stt"]["inputs"]["min-p"] = gr.Slider(value=0.0, minimum=0.0, maximum=1.0, step=0.05, label="Min P")
							layout["inference_stt"]["inputs"]["beam-width"] = gr.Slider(value=0, minimum=0, maximum=32, step=1, label="Beam Width", info="Number of branches to search through for beam search sampling.")
						with gr.Row():
							layout["inference_stt"]["inputs"]["repetition-penalty"] = gr.Slider(value=1.0, minimum=-2.0, maximum=2.0, step=0.05, label="Repetition Penalty", info="Incurs a penalty to tokens based on how often they appear in a sequence.")
							layout["inference_stt"]["inputs"]["repetition-penalty-decay"] = gr.Slider(value=0.0, minimum=-2.0, maximum=2.0, step=0.05, label="Repetition Penalty Length Decay", info="Modifies the reptition penalty based on how far back in time the token appeared in the sequence.")
							layout["inference_stt"]["inputs"]["length-penalty"] = gr.Slider(value=0.0, minimum=-2.0, maximum=2.0, step=0.05, label="Length Penalty", info="(AR only) Modifies the probability of a stop token based on the current length of the sequence.")
						"""
						with gr.Row():
							layout["inference_stt"]["inputs"]["dynamic-sampling"] = gr.Checkbox(label="Dynamic Temperature", info="Dynamically adjusts the temperature based on the highest confident predicted token per sampling step.")
							layout["inference_stt"]["inputs"]["mirostat-tau"] = gr.Slider(value=0.0, minimum=0.0, maximum=8.0, step=0.05, label="Mirostat τ (Tau)", info="The \"surprise\" value when performing mirostat sampling. 0 to disable.")
							layout["inference_stt"]["inputs"]["mirostat-eta"] = gr.Slider(value=0.0, minimum=0.0, maximum=2.0, step=0.05, label="Mirostat η (Eta)", info="The \"learning rate\" during mirostat sampling applied to the maximum surprise.")
						with gr.Row():
							layout["inference_stt"]["inputs"]["dry-multiplier"] = gr.Slider(value=0.0, minimum=0.0, maximum=8.0, step=0.05, label="DRY Multiplier", info="The multiplying factor for the DRY score penalty (0 to disable DRY sampling).")
							layout["inference_stt"]["inputs"]["dry-base"] = gr.Slider(value=1.75, minimum=0.0, maximum=8.0, step=0.05, label="DRY Base", info="The base of the exponent in the DRY score penalty")
							layout["inference_stt"]["inputs"]["dry-allowed-length"] = gr.Slider(value=2, minimum=0, maximum=75, step=1, label="Allowed Length", info="The maximimum length a token can be to perform DRY penalty with.")
						"""

		layout["inference_stt"]["buttons"]["inference"].click(
			fn=do_inference_stt,
			inputs=[ x for x in layout["inference_stt"]["inputs"].values() if x is not None],
			outputs=[ x for x in layout["inference_stt"]["outputs"].values() if x is not None]
		)

		
	"""
	with gr.Tab("Training"):
		with gr.Row():
			with gr.Column(scale=1):
				layout["training"]["outputs"]["console"] = gr.Textbox(lines=8, label="Console Log")
		with gr.Row():
			with gr.Column(scale=1):
				layout["training"]["buttons"]["train"] = gr.Button(value="Train")

		layout["training"]["buttons"]["train"].click(
			fn=do_training,
			outputs=[ x for x in layout["training"]["outputs"].values() if x is not None],
		)
	"""

	if not USING_SPACES:
		with gr.Tab("Dataset"):
			with gr.Row():
				with gr.Column(scale=7):
					layout["dataset"]["outputs"]["transcription"] = gr.Textbox(lines=5, label="Sample Metadata")
				with gr.Column(scale=1):
					layout["dataset"]["inputs"]["speaker"] = gr.Dropdown(choices=get_speakers(), label="Speakers")
					layout["dataset"]["outputs"]["audio"] = gr.Audio(label="Output")
					layout["dataset"]["buttons"]["sample"] = gr.Button(value="Sample")

				layout["dataset"]["buttons"]["sample"].click(
					fn=load_sample,
					inputs=[ x for x in layout["dataset"]["inputs"].values() if x is not None],
					outputs=[ x for x in layout["dataset"]["outputs"].values() if x is not None],
				)

	if not USING_SPACES:
		with gr.Tab("Settings"):
			with gr.Row():
				with gr.Column(scale=1):
					layout["settings"]["buttons"]["load"] = gr.Button(value="Load Model")
				with gr.Column(scale=7):
					with gr.Row():
						layout["settings"]["inputs"]["models"] = gr.Dropdown(choices=get_model_paths(), value=args.yaml or args.model, label="Model", info="Model to load. Can load from a config YAML or the weights itself.")
						layout["settings"]["inputs"]["device"] = gr.Dropdown(choices=get_devices(), value="cuda:0", label="Device", info="Device to load the weights onto.")
					with gr.Row():
						layout["settings"]["inputs"]["dtype"] = gr.Dropdown(choices=get_dtypes(), value="auto", label="Precision", info="Tensor type to load the model under.")
						layout["settings"]["inputs"]["attentions"] = gr.Dropdown(choices=get_attentions(), value="auto", label="Attentions", info="Attention mechanism to utilize.")

				layout["settings"]["buttons"]["load"].click(
					fn=load_model,
					inputs=[ x for x in layout["settings"]["inputs"].values() if x is not None],
					outputs=[ x for x in layout["settings"]["outputs"].values() if x is not None],
				)

	if os.path.exists("README.md") and args.render_markdown:
		md = open("README.md", "r", encoding="utf-8").read()
		# remove HF's metadata
		if md.startswith("---\n"):
			md = "".join(md.split("---")[2:])
		gr.Markdown(md)

def start( lock=True ):
	setup_logging()

	if not USING_SPACES:
		ui.queue(max_size=8)
		ui.launch(share=args.share, server_name=args.listen_host, server_port=args.listen_port, prevent_thread_lock=not lock)
	else:
		ui.queue().launch()

if __name__ == "__main__":
	start()