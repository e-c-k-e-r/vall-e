import os
import re
import argparse
import random
import tempfile
import functools
from datetime import datetime

import gradio as gr

from time import perf_counter
from pathlib import Path

from .inference import TTS, cfg
from .train import train

tts = None

layout = {}
layout["inference"] = {}
layout["training"] = {}

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
			return fun(**kwargs)
		return wrapped_function
	return decorated

class timer:
    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        print(f'[{datetime.now().isoformat()}] Elapsed time: {(perf_counter() - self.start):.3f}s')

def init_tts(restart=False):
	global tts

	if tts is not None:
		if not restart:
			return tts
		del tts
	
	parser = argparse.ArgumentParser(allow_abbrev=False)
	parser.add_argument("--yaml", type=Path, default=os.environ.get('VALLE_YAML', None)) # os environ so it can be specified in a HuggingFace Space too
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--amp", action="store_true")
	parser.add_argument("--dtype", type=str, default="auto")
	args, unknown = parser.parse_known_args()

	tts = TTS( config=args.yaml, device=args.device, dtype=args.dtype if args.dtype != "auto" else None, amp=args.amp )
	return tts

@gradio_wrapper(inputs=layout["inference"]["inputs"].keys())
def do_inference( progress=gr.Progress(track_tqdm=True), *args, **kwargs ):
	if kwargs.pop("dynamic-sampling", False):
		kwargs['min-ar-temp'] = 0.85 if kwargs['ar-temp'] > 0.85 else 0.0
		kwargs['min-nar-temp'] = 0.85 if kwargs['nar-temp'] > 0.85 else 0.0 # should probably disable it for the NAR
	else:
		kwargs['min-ar-temp'] = -1
		kwargs['min-nar-temp'] = -1

	parser = argparse.ArgumentParser(allow_abbrev=False)
	# I'm very sure I can procedurally generate this list
	parser.add_argument("--text", type=str, default=kwargs["text"])
	parser.add_argument("--references", type=str, default=kwargs["reference"])
	parser.add_argument("--language", type=str, default="en")
	parser.add_argument("--input-prompt-length", type=float, default=kwargs["input-prompt-length"])
	parser.add_argument("--max-ar-steps", type=int, default=int(kwargs["max-seconds"]*cfg.dataset.frames_per_second))
	parser.add_argument("--max-ar-context", type=int, default=int(kwargs["max-seconds-context"]*cfg.dataset.frames_per_second))
	parser.add_argument("--max-nar-levels", type=int, default=kwargs["max-nar-levels"])
	parser.add_argument("--ar-temp", type=float, default=kwargs["ar-temp"])
	parser.add_argument("--nar-temp", type=float, default=kwargs["nar-temp"])
	parser.add_argument("--min-ar-temp", type=float, default=kwargs["min-ar-temp"])
	parser.add_argument("--min-nar-temp", type=float, default=kwargs["min-nar-temp"])
	parser.add_argument("--top-p", type=float, default=kwargs["top-p"])
	parser.add_argument("--top-k", type=int, default=kwargs["top-k"])
	parser.add_argument("--repetition-penalty", type=float, default=kwargs["repetition-penalty"])
	parser.add_argument("--repetition-penalty-decay", type=float, default=kwargs["repetition-penalty-decay"])
	parser.add_argument("--length-penalty", type=float, default=kwargs["length-penalty"])
	parser.add_argument("--beam-width", type=int, default=kwargs["beam-width"])
	parser.add_argument("--mirostat-tau", type=float, default=kwargs["mirostat-tau"])
	parser.add_argument("--mirostat-eta", type=float, default=kwargs["mirostat-eta"])
	args, unknown = parser.parse_known_args()

	tmp = tempfile.NamedTemporaryFile(suffix='.wav')

	if not args.references:
		raise ValueError("No reference audio provided.")

	tts = init_tts()
	with timer() as t:
		wav, sr = tts.inference(
			text=args.text,
			language=args.language,
			references=[args.references.split(";")],
			out_path=tmp.name,
			max_ar_steps=args.max_ar_steps,
			max_nar_levels=args.max_nar_levels,
			input_prompt_length=args.input_prompt_length,
			ar_temp=args.ar_temp,
			nar_temp=args.nar_temp,
			min_ar_temp=args.min_ar_temp,
			min_nar_temp=args.min_nar_temp,
			top_p=args.top_p,
			top_k=args.top_k,
			repetition_penalty=args.repetition_penalty,
			repetition_penalty_decay=args.repetition_penalty_decay,
			length_penalty=args.length_penalty,
			mirostat_tau=args.mirostat_tau,
			mirostat_eta=args.mirostat_eta,
		)
	
	wav = wav.squeeze(0).cpu().numpy()
	return (sr, wav)

"""
@gradio_wrapper(inputs=layout["training"]["inputs"].keys())
def do_training( progress=gr.Progress(track_tqdm=True), *args, **kwargs ):
	while True:
		metrics = next(it)
		yield metrics
"""

def get_random_prompt():
	harvard_sentences=[
		"The birch canoe slid on the smooth planks.",
		"Glue the sheet to the dark blue background.",
		"It's easy to tell the depth of a well.",
		"These days a chicken leg is a rare dish.",
		"Rice is often served in round bowls.",
		"The juice of lemons makes fine punch.",
		"The box was thrown beside the parked truck.",
		"The hogs were fed chopped corn and garbage.",
		"Four hours of steady work faced us.",
		"A large size in stockings is hard to sell.",
		"The boy was there when the sun rose.",
		"A rod is used to catch pink salmon.",
		"The source of the huge river is the clear spring.",
		"Kick the ball straight and follow through.",
		"Help the woman get back to her feet.",
		"A pot of tea helps to pass the evening.",
		"Smoky fires lack flame and heat.",
		"The soft cushion broke the man's fall.",
		"The salt breeze came across from the sea.",
		"The girl at the booth sold fifty bonds.",
		"The small pup gnawed a hole in the sock.",
		"The fish twisted and turned on the bent hook.",
		"Press the pants and sew a button on the vest.",
		"The swan dive was far short of perfect.",
		"The beauty of the view stunned the young boy.",
		"Two blue fish swam in the tank.",
		"Her purse was full of useless trash.",
		"The colt reared and threw the tall rider.",
		"It snowed, rained, and hailed the same morning.",
		"Read verse out loud for pleasure.",
	]
	return random.choice(harvard_sentences)

# setup args
parser = argparse.ArgumentParser(allow_abbrev=False)
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
		with gr.Row():
			with gr.Column(scale=8):
				layout["inference"]["inputs"]["text"] = gr.Textbox(lines=5, value=get_random_prompt, label="Input Prompt")
		with gr.Row():
			with gr.Column(scale=1):
				layout["inference"]["inputs"]["reference"] = gr.Audio(label="Audio Input", sources=["upload"], type="filepath") #, info="Reference audio for TTS")
				# layout["inference"]["stop"] = gr.Button(value="Stop")
				layout["inference"]["outputs"]["output"] = gr.Audio(label="Output")
				layout["inference"]["buttons"]["inference"] = gr.Button(value="Inference")
			with gr.Column(scale=7):
				with gr.Row():
					layout["inference"]["inputs"]["max-seconds"] = gr.Slider(value=12, minimum=1, maximum=32, step=0.1, label="Maximum Seconds", info="Limits how many steps to perform in the AR pass.")
					layout["inference"]["inputs"]["max-nar-levels"] = gr.Slider(value=7, minimum=0, maximum=7, step=1, label="Max NAR Levels", info="Limits how many steps to perform in the NAR pass.")
					layout["inference"]["inputs"]["input-prompt-length"] = gr.Slider(value=3.0, minimum=0.0, maximum=12.0, step=0.05, label="Input Prompt Trim Length", info="Trims the input prompt down to X seconds. Set 0 to disable.")
					layout["inference"]["inputs"]["max-seconds-context"] = gr.Slider(value=0.0, minimum=0.0, maximum=12.0, step=0.05, label="Context Length", info="Amount of generated audio to keep in the context during inference, in seconds. Set 0 to disable.")
				with gr.Row():
					layout["inference"]["inputs"]["ar-temp"] = gr.Slider(value=0.95, minimum=0.0, maximum=1.5, step=0.05, label="Temperature (AR)", info="Modifies the randomness from the samples in the AR. (0 to greedy sample)")
					layout["inference"]["inputs"]["nar-temp"] = gr.Slider(value=0.01, minimum=0.0, maximum=1.5, step=0.05, label="Temperature (NAR)", info="Modifies the randomness from the samples in the NAR. (0 to greedy sample)")
				with gr.Row():
					layout["inference"]["inputs"]["dynamic-sampling"] = gr.Checkbox(label="Dynamic Temperature", info="Dynamically adjusts the temperature based on the highest confident predicted token per sampling step.")

				with gr.Row():
					layout["inference"]["inputs"]["top-p"] = gr.Slider(value=1.0, minimum=0.0, maximum=1.0, step=0.05, label="Top P", info=r"Limits the samples that are outside the top P% of probabilities.")
					layout["inference"]["inputs"]["top-k"] = gr.Slider(value=0, minimum=0, maximum=1024, step=1, label="Top K", info="Limits the samples to the top K of probabilities.")
					layout["inference"]["inputs"]["beam-width"] = gr.Slider(value=0, minimum=0, maximum=32, step=1, label="Beam Width", info="Number of branches to search through for beam search sampling.")
				with gr.Row():
					layout["inference"]["inputs"]["repetition-penalty"] = gr.Slider(value=1.0, minimum=-2.0, maximum=2.0, step=0.05, label="Repetition Penalty", info="Incurs a penalty to tokens based on how often they appear in a sequence.")
					layout["inference"]["inputs"]["repetition-penalty-decay"] = gr.Slider(value=0.0, minimum=-2.0, maximum=2.0, step=0.05, label="Repetition Penalty Length Decay", info="Modifies the reptition penalty based on how far back in time the token appeared in the sequence.")
					layout["inference"]["inputs"]["length-penalty"] = gr.Slider(value=0.0, minimum=-2.0, maximum=2.0, step=0.05, label="Length Penalty", info="(AR only) Modifies the probability of a stop token based on the current length of the sequence.")
				with gr.Row():
					layout["inference"]["inputs"]["mirostat-tau"] = gr.Slider(value=0.0, minimum=0.0, maximum=8.0, step=0.05, label="Mirostat τ (Tau)", info="The \"surprise\" value when performing mirostat sampling. 0 to disable.")
					layout["inference"]["inputs"]["mirostat-eta"] = gr.Slider(value=0.0, minimum=0.0, maximum=2.0, step=0.05, label="Mirostat η (Eta)", info="The \"learning rate\" during mirostat sampling applied to the maximum surprise.")

		layout["inference"]["buttons"]["inference"].click(
			fn=do_inference,
			inputs=[ x for x in layout["inference"]["inputs"].values() if x is not None],
			outputs=[ x for x in layout["inference"]["outputs"].values() if x is not None]
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

	if os.path.exists("README.md") and args.render_markdown:
		md = open("README.md", "r", encoding="utf-8").read()
		# remove HF's metadata
		if md.startswith("---\n"):
			md = "".join(md.split("---")[2:])
		gr.Markdown(md)

def start( lock=True ):
	ui.queue(max_size=8)
	ui.launch(share=args.share, server_name=args.listen_host, server_port=args.listen_port, prevent_thread_lock=not lock)

if __name__ == "__main__":
	start()