import os
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

from .inference import TTS, cfg
from .train import train
from .utils import get_devices, setup_logging, timer
from .utils.io import json_read, json_stringify
from .emb.qnt import decode_to_wave
from .data import get_lang_symmap, get_random_prompt

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

# returns a list of models, assuming the models are placed under ./training/ or ./models/
def get_model_paths( paths=[Path("./training/"), Path("./models/")] ):
	yamls = []

	for path in paths:
		if not path.exists():
			continue

		for yaml in path.glob("**/*.yaml"):
			if "/logs/" in str(yaml):
				continue

			yamls.append( yaml )

	return yamls

def get_dtypes():
	return ["float32", "float16", "bfloat16", "float8_e5m2", "float8_e4m3fn", "auto"]

from .models.arch import AVAILABLE_ATTENTIONS
def get_attentions():
	return AVAILABLE_ATTENTIONS + ["auto"]

#@gradio_wrapper(inputs=layout["settings"]["inputs"].keys())
def load_model( yaml, device, dtype, attention ):
	gr.Info(f"Loading: {yaml}")
	try:
		init_tts( yaml=Path(yaml), restart=True, device=device, dtype=dtype, attention=attention )
	except Exception as e:
		raise gr.Error(e)
	gr.Info(f"Loaded model")

def get_speakers():
	return cfg.dataset.training

def get_languages():
	return get_lang_symmap().keys()

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

def init_tts(yaml=None, restart=False, device="cuda", dtype="auto", attention=None):
	global tts

	if tts is not None:
		if not restart:
			return tts
		
		del tts
		tts = None
	
	parser = argparse.ArgumentParser(allow_abbrev=False)
	parser.add_argument("--yaml", type=Path, default=os.environ.get('VALLE_YAML', yaml)) # os environ so it can be specified in a HuggingFace Space too
	parser.add_argument("--device", type=str, default=device)
	parser.add_argument("--amp", action="store_true")
	parser.add_argument("--dtype", type=str, default=dtype)
	parser.add_argument("--attention", type=str, default=attention)
	args, unknown = parser.parse_known_args()

	tts = TTS( config=args.yaml if yaml is None else yaml, device=args.device, dtype=args.dtype if args.dtype != "auto" else None, amp=args.amp, attention=args.attention )
	return tts

@gradio_wrapper(inputs=layout["inference_tts"]["inputs"].keys())
def do_inference_tts( progress=gr.Progress(track_tqdm=True), *args, **kwargs ):
	if not cfg.yaml_path:
		raise Exception("No YAML loaded.")

	if kwargs.pop("dynamic-sampling", False):
		kwargs['min-ar-temp'] = 0.85 if kwargs['ar-temp'] > 0.85 else 0.0
		kwargs['min-nar-temp'] = 0.85 if kwargs['nar-temp'] > 0.85 else 0.0 # should probably disable it for the NAR
	else:
		kwargs['min-ar-temp'] = -1
		kwargs['min-nar-temp'] = -1

	parser = argparse.ArgumentParser(allow_abbrev=False)
	# I'm very sure I can procedurally generate this list
	parser.add_argument("--text", type=str, default=kwargs["text"])
	parser.add_argument("--task", type=str, default="tts")
	parser.add_argument("--references", type=str, default=kwargs["reference"])
	parser.add_argument("--language", type=str, default=kwargs["language"])
	parser.add_argument("--input-prompt-length", type=float, default=kwargs["input-prompt-length"])
	#parser.add_argument("--input-prompt-prefix", action='store_true', default=kwargs["input-prompt-prefix"])
	parser.add_argument("--input-prompt-prefix", action='store_true')
	parser.add_argument("--max-ar-steps", type=int, default=int(kwargs["max-seconds"]*cfg.dataset.frames_per_second))
	parser.add_argument("--max-nar-levels", type=int, default=0), # kwargs["max-nar-levels"])
	parser.add_argument("--ar-temp", type=float, default=kwargs["ar-temp"])
	parser.add_argument("--nar-temp", type=float, default=kwargs["nar-temp"])
	parser.add_argument("--min-ar-temp", type=float, default=kwargs["min-ar-temp"])
	parser.add_argument("--min-nar-temp", type=float, default=kwargs["min-nar-temp"])
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
	parser.add_argument("--entropix-sampling", action="store_true")
	args, unknown = parser.parse_known_args()

	tmp = tempfile.NamedTemporaryFile(suffix='.wav')

	"""
	if not args.references:
		raise Exception("No reference audio provided.")
	"""

	if kwargs.pop("entropix-sampling", False):
		args.entropix_sampling = True

	tts = init_tts()
	
	gr.Info("Inferencing...")
	
	with timer("Inferenced in", callback=lambda msg: gr.Info( msg )) as t:
		wav, sr = tts.inference(
			text=args.text,
			language=args.language,
			task=args.task,
			references=args.references.split(";") if args.references is not None else [],
			out_path=tmp.name,
			max_ar_steps=args.max_ar_steps,
			max_nar_levels=args.max_nar_levels,
			input_prompt_length=args.input_prompt_length,
			input_prompt_prefix=args.input_prompt_prefix,
			ar_temp=args.ar_temp,
			nar_temp=args.nar_temp,
			min_ar_temp=args.min_ar_temp,
			min_nar_temp=args.min_nar_temp,
			top_p=args.top_p,
			top_k=args.top_k,
			min_p=args.min_p,
			repetition_penalty=args.repetition_penalty,
			repetition_penalty_decay=args.repetition_penalty_decay,
			length_penalty=args.length_penalty,
			mirostat_tau=args.mirostat_tau,
			mirostat_eta=args.mirostat_eta,
			dry_multiplier=args.dry_multiplier,
			dry_base=args.dry_base,
			dry_allowed_length=args.dry_allowed_length,
			entropix_sampling=args.entropix_sampling
		)
	
	wav = wav.squeeze(0).cpu().numpy()
	return (sr, wav)

@gradio_wrapper(inputs=layout["inference_stt"]["inputs"].keys())
def do_inference_stt( progress=gr.Progress(track_tqdm=True), *args, **kwargs ):
	if not cfg.yaml_path:
		raise Exception("No YAML loaded.")

	if kwargs.pop("dynamic-sampling", False):
		kwargs['min-ar-temp'] = 0.85 if kwargs['ar-temp'] > 0.85 else 0.0
	else:
		kwargs['min-ar-temp'] = -1

	parser = argparse.ArgumentParser(allow_abbrev=False)
	# I'm very sure I can procedurally generate this list
	parser.add_argument("--references", type=str, default=kwargs["reference"])
	parser.add_argument("--language", type=str, default=kwargs["language"])
	parser.add_argument("--max-ar-steps", type=int, default=0)
	parser.add_argument("--ar-temp", type=float, default=kwargs["ar-temp"])
	parser.add_argument("--min-ar-temp", type=float, default=kwargs["min-ar-temp"])
	parser.add_argument("--top-p", type=float, default=kwargs["top-p"])
	parser.add_argument("--top-k", type=int, default=kwargs["top-k"])
	parser.add_argument("--min-p", type=int, default=kwargs["min-p"])
	parser.add_argument("--repetition-penalty", type=float, default=kwargs["repetition-penalty"])
	parser.add_argument("--repetition-penalty-decay", type=float, default=kwargs["repetition-penalty-decay"])
	parser.add_argument("--length-penalty", type=float, default=kwargs["length-penalty"])
	parser.add_argument("--beam-width", type=int, default=kwargs["beam-width"])
	parser.add_argument("--mirostat-tau", type=float, default=kwargs["mirostat-tau"])
	parser.add_argument("--mirostat-eta", type=float, default=kwargs["mirostat-eta"])
	parser.add_argument("--dry-multiplier", type=float, default=kwargs["dry-multiplier"])
	parser.add_argument("--dry-base", type=float, default=kwargs["dry-base"])
	parser.add_argument("--dry-allowed-length", type=int, default=kwargs["dry-allowed-length"])
	parser.add_argument("--entropix-sampling", action="store_true")
	args, unknown = parser.parse_known_args()


	"""
	if not args.references:
		raise Exception("No reference audio provided.")
	"""

	args.references = args.references.split(";") if args.references is not None else []
	if args.max_ar_steps == 0:
		for i, path in enumerate( args.references ):
			metadata = torchaudio.info(path)
			duration = metadata.num_frames / metadata.sample_rate
			args.max_ar_steps += duration
		args.max_ar_steps = math.floor( args.max_ar_steps * 20 ) # assume 20 tokens per second
	
	if kwargs.pop("entropix-sampling", False):
		args.entropix_sampling = True

	tts = init_tts()
	
	gr.Info("Inferencing...")
	with timer("Inferenced in") as t:
		text = tts.inference(
			text="",
			language=args.language,
			task="stt",
			references=args.references,
			max_ar_steps=args.max_ar_steps,
			ar_temp=args.ar_temp,
			min_ar_temp=args.min_ar_temp,
			top_p=args.top_p,
			top_k=args.top_k,
			min_p=args.min_p,
			repetition_penalty=args.repetition_penalty,
			repetition_penalty_decay=args.repetition_penalty_decay,
			length_penalty=args.length_penalty,
			mirostat_tau=args.mirostat_tau,
			mirostat_eta=args.mirostat_eta,
			dry_multiplier=args.dry_multiplier,
			dry_base=args.dry_base,
			dry_allowed_length=args.dry_allowed_length,
			entropix_sampling=args.entropix_sampling,
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
					layout["inference_tts"]["inputs"]["text"] = gr.Textbox(lines=5, value=get_random_prompt, label="Input Prompt")
			with gr.Row():
				with gr.Column(scale=1):
					layout["inference_tts"]["inputs"]["reference"] = gr.Audio(label="Audio Input", sources=["upload"], type="filepath") #, info="Reference audio for TTS")
					# layout["inference_tts"]["stop"] = gr.Button(value="Stop")
					layout["inference_tts"]["outputs"]["output"] = gr.Audio(label="Output")
					layout["inference_tts"]["buttons"]["inference"] = gr.Button(value="Inference")
				with gr.Column(scale=7):
					with gr.Tab("Basic Settings"):
						with gr.Row():
							layout["inference_tts"]["inputs"]["max-seconds"] = gr.Slider(value=12, minimum=1, maximum=32, step=0.1, label="Maximum Seconds", info="Limits how many steps to perform in the AR pass.")
							#layout["inference_tts"]["inputs"]["max-nar-levels"] = gr.Slider(value=7, minimum=0, maximum=7, step=1, label="Max NAR Levels", info="Limits how many steps to perform in the NAR pass.")
							layout["inference_tts"]["inputs"]["input-prompt-length"] = gr.Slider(value=5.0, minimum=0.0, maximum=12.0, step=0.05, label="Input Prompt Trim Length", info="Trims the input prompt down to X seconds. Set 0 to disable.")
						with gr.Row():
							layout["inference_tts"]["inputs"]["ar-temp"] = gr.Slider(value=0.9, minimum=0.0, maximum=1.5, step=0.05, label="Temperature (AR)", info="Modifies the randomness from the samples in the AR. (0 to greedy sample)")
							layout["inference_tts"]["inputs"]["nar-temp"] = gr.Slider(value=0.0, minimum=0.0, maximum=1.5, step=0.05, label="Temperature (NAR)", info="Modifies the randomness from the samples in the NAR. (0 to greedy sample)")
						with gr.Row():
							#layout["inference_tts"]["inputs"]["input-prompt-prefix"] = gr.Checkbox(label="Input Prompt as Prefix", info="Treats the input prompt clip as the prefix of the generated sequence.")
							layout["inference_tts"]["inputs"]["dynamic-sampling"] = gr.Checkbox(label="Dynamic Temperature", info="Dynamically adjusts the temperature based on the highest confident predicted token per sampling step.")
							layout["inference_tts"]["inputs"]["entropix-sampling"] = gr.Checkbox(label="Entropix Sampling", info="Dynamically samples based on entropy/varentropy values from the logits / attention scores.")
							layout["inference_tts"]["inputs"]["language"] = gr.Dropdown(choices=get_languages(), label="Language", value="en")
					with gr.Tab("Sampler Settings"):
						with gr.Row():
							layout["inference_tts"]["inputs"]["top-p"] = gr.Slider(value=1.0, minimum=0.0, maximum=1.0, step=0.05, label="Top P", info=r"Limits the samples that are outside the top P% of probabilities.")
							layout["inference_tts"]["inputs"]["top-k"] = gr.Slider(value=0, minimum=0, maximum=1024, step=1, label="Top K", info="Limits the samples to the top K of probabilities.")
							layout["inference_tts"]["inputs"]["min-p"] = gr.Slider(value=0.0, minimum=0.0, maximum=1.0, step=0.05, label="Min P")
							layout["inference_tts"]["inputs"]["beam-width"] = gr.Slider(value=0, minimum=0, maximum=32, step=1, label="Beam Width", info="Number of branches to search through for beam search sampling.")
						with gr.Row():
							layout["inference_tts"]["inputs"]["repetition-penalty"] = gr.Slider(value=1.0, minimum=-2.0, maximum=2.0, step=0.05, label="Repetition Penalty", info="Incurs a penalty to tokens based on how often they appear in a sequence.")
							layout["inference_tts"]["inputs"]["repetition-penalty-decay"] = gr.Slider(value=0.0, minimum=-2.0, maximum=2.0, step=0.05, label="Repetition Penalty Length Decay", info="Modifies the reptition penalty based on how far back in time the token appeared in the sequence.")
							layout["inference_tts"]["inputs"]["length-penalty"] = gr.Slider(value=0.0, minimum=-2.0, maximum=2.0, step=0.05, label="Length Penalty", info="(AR only) Modifies the probability of a stop token based on the current length of the sequence.")
						with gr.Row():
							layout["inference_tts"]["inputs"]["mirostat-tau"] = gr.Slider(value=0.0, minimum=0.0, maximum=8.0, step=0.05, label="Mirostat τ (Tau)", info="The \"surprise\" value when performing mirostat sampling. 0 to disable.")
							layout["inference_tts"]["inputs"]["mirostat-eta"] = gr.Slider(value=0.0, minimum=0.0, maximum=2.0, step=0.05, label="Mirostat η (Eta)", info="The \"learning rate\" during mirostat sampling applied to the maximum surprise.")
						with gr.Row():
							layout["inference_tts"]["inputs"]["dry-multiplier"] = gr.Slider(value=0.0, minimum=0.0, maximum=8.0, step=0.05, label="DRY Multiplier", info="The multiplying factor for the DRY score penalty (0 to disable DRY sampling).")
							layout["inference_tts"]["inputs"]["dry-base"] = gr.Slider(value=1.75, minimum=0.0, maximum=8.0, step=0.05, label="DRY Base", info="The base of the exponent in the DRY score penalty")
							layout["inference_tts"]["inputs"]["dry-allowed-length"] = gr.Slider(value=2, minimum=0, maximum=75, step=1, label="Allowed Length", info="The maximimum length a token can be to perform DRY penalty with.")

		layout["inference_tts"]["buttons"]["inference"].click(
			fn=do_inference_tts,
			inputs=[ x for x in layout["inference_tts"]["inputs"].values() if x is not None],
			outputs=[ x for x in layout["inference_tts"]["outputs"].values() if x is not None]
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
							layout["inference_stt"]["inputs"]["ar-temp"] = gr.Slider(value=0.0, minimum=0.0, maximum=1.5, step=0.05, label="Temperature (AR)", info="Modifies the randomness from the samples in the AR. (0 to greedy sample)")
						with gr.Row():
							layout["inference_stt"]["inputs"]["dynamic-sampling"] = gr.Checkbox(label="Dynamic Temperature", info="Dynamically adjusts the temperature based on the highest confident predicted token per sampling step.")
							layout["inference_stt"]["inputs"]["entropix-sampling"] = gr.Checkbox(label="Entropix Sampling", info="Dynamically samples based on entropy/varentropy values from the logits / attention scores.")
							layout["inference_stt"]["inputs"]["language"] = gr.Dropdown(choices=get_languages(), label="Language", value="en")
					with gr.Tab("Sampler Settings"):
						with gr.Row():
							layout["inference_stt"]["inputs"]["top-p"] = gr.Slider(value=1.0, minimum=0.0, maximum=1.0, step=0.05, label="Top P", info=r"Limits the samples that are outside the top P% of probabilities.")
							layout["inference_stt"]["inputs"]["top-k"] = gr.Slider(value=0, minimum=0, maximum=1024, step=1, label="Top K", info="Limits the samples to the top K of probabilities.")
							layout["inference_stt"]["inputs"]["min-p"] = gr.Slider(value=0.0, minimum=0.0, maximum=1.0, step=0.05, label="Min P")
							layout["inference_stt"]["inputs"]["beam-width"] = gr.Slider(value=0, minimum=0, maximum=32, step=1, label="Beam Width", info="Number of branches to search through for beam search sampling.")
						with gr.Row():
							layout["inference_stt"]["inputs"]["repetition-penalty"] = gr.Slider(value=1.25, minimum=-2.0, maximum=2.0, step=0.05, label="Repetition Penalty", info="Incurs a penalty to tokens based on how often they appear in a sequence.")
							layout["inference_stt"]["inputs"]["repetition-penalty-decay"] = gr.Slider(value=0.0, minimum=-2.0, maximum=2.0, step=0.05, label="Repetition Penalty Length Decay", info="Modifies the reptition penalty based on how far back in time the token appeared in the sequence.")
							layout["inference_stt"]["inputs"]["length-penalty"] = gr.Slider(value=0.0, minimum=-2.0, maximum=2.0, step=0.05, label="Length Penalty", info="(AR only) Modifies the probability of a stop token based on the current length of the sequence.")
						with gr.Row():
							layout["inference_stt"]["inputs"]["mirostat-tau"] = gr.Slider(value=0.0, minimum=0.0, maximum=8.0, step=0.05, label="Mirostat τ (Tau)", info="The \"surprise\" value when performing mirostat sampling. 0 to disable.")
							layout["inference_stt"]["inputs"]["mirostat-eta"] = gr.Slider(value=0.0, minimum=0.0, maximum=2.0, step=0.05, label="Mirostat η (Eta)", info="The \"learning rate\" during mirostat sampling applied to the maximum surprise.")
						with gr.Row():
							layout["inference_stt"]["inputs"]["dry-multiplier"] = gr.Slider(value=0.0, minimum=0.0, maximum=8.0, step=0.05, label="DRY Multiplier", info="The multiplying factor for the DRY score penalty (0 to disable DRY sampling).")
							layout["inference_stt"]["inputs"]["dry-base"] = gr.Slider(value=1.75, minimum=0.0, maximum=8.0, step=0.05, label="DRY Base", info="The base of the exponent in the DRY score penalty")
							layout["inference_stt"]["inputs"]["dry-allowed-length"] = gr.Slider(value=2, minimum=0, maximum=75, step=1, label="Allowed Length", info="The maximimum length a token can be to perform DRY penalty with.")

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

	with gr.Tab("Settings"):
		with gr.Row():
			with gr.Column(scale=7):
				with gr.Row():
					layout["settings"]["inputs"]["models"] = gr.Dropdown(choices=get_model_paths(), value=args.yaml, label="Model")
					layout["settings"]["inputs"]["device"] = gr.Dropdown(choices=get_devices(), value="cuda:0", label="Device")
					layout["settings"]["inputs"]["dtype"] = gr.Dropdown(choices=get_dtypes(), value="auto", label="Precision")
					layout["settings"]["inputs"]["attentions"] = gr.Dropdown(choices=get_attentions(), value="auto", label="Attentions")
			with gr.Column(scale=1):
				layout["settings"]["buttons"]["load"] = gr.Button(value="Load Model")

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

	ui.queue(max_size=8)
	ui.launch(share=args.share, server_name=args.listen_host, server_port=args.listen_port, prevent_thread_lock=not lock)

if __name__ == "__main__":
	start()