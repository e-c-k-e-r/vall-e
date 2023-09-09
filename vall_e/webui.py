import os
import re
import argparse
import tempfile
import functools

import gradio as gr

from pathlib import Path

from .inference import TTS

tts = None

layout = {}
layout["inference"] = {}
layout["inference"]["inputs"] = {
	"progress": None
}
layout["inference"]["outputs"] = {}
layout["inference"]["buttons"] = {}

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

def init_tts(restart=False):
	global tts

	if tts is not None:
		if not restart:
			return tts
		del tts
	
	parser = argparse.ArgumentParser(allow_abbrev=False)
	parser.add_argument("--yaml", type=Path, default=os.environ.get('VALLE_YAML', None)) # os environ so it can be specified in a HuggingFace Space too
	parser.add_argument("--ar-ckpt", type=Path, default=None)
	parser.add_argument("--nar-ckpt", type=Path, default=None)
	parser.add_argument("--device", type=str, default="cpu")
	parser.add_argument("--amp", action="store_true")
	parser.add_argument("--dtype", type=str, default="float32")
	args, unknown = parser.parse_known_args()

	tts = TTS( config=args.yaml, ar_ckpt=args.ar_ckpt, nar_ckpt=args.nar_ckpt, device=args.device, dtype=args.dtype, amp=args.amp )
	return tts

@gradio_wrapper(inputs=layout["inference"]["inputs"].keys())
def do_inference( progress=gr.Progress(track_tqdm=True), *args, **kwargs ):
	parser = argparse.ArgumentParser(allow_abbrev=False)
	parser.add_argument("--text", type=str, default=kwargs["text"])
	parser.add_argument("--references", type=str, default=kwargs["reference"])
	parser.add_argument("--max-ar-steps", type=int, default=kwargs["steps"])
	parser.add_argument("--ar-temp", type=float, default=kwargs["ar-temp"])
	parser.add_argument("--nar-temp", type=float, default=kwargs["nar-temp"])
	parser.add_argument("--top-p", type=float, default=1.0)
	parser.add_argument("--top-k", type=int, default=0)
	parser.add_argument("--repetition-penalty", type=float, default=1.0)
	parser.add_argument("--repetition-penalty-decay", type=float, default=0.0)
	parser.add_argument("--length-penalty", type=float, default=0.0)
	args, unknown = parser.parse_known_args()

	tmp = tempfile.NamedTemporaryFile(suffix='.wav')

	tts = init_tts()
	wav, sr = tts.inference(
		text=args.text,
		references=[args.references.split(";")],
		out_path=tmp.name,
		max_ar_steps=args.max_ar_steps,
		ar_temp=args.ar_temp,
		nar_temp=args.nar_temp,
		top_p=args.top_p,
		top_k=args.top_k,
		repetition_penalty=args.repetition_penalty,
		repetition_penalty_decay=args.repetition_penalty_decay,
		length_penalty=args.length_penalty
	)
	
	wav = wav.squeeze(0).cpu().numpy()
	return (sr, wav)

ui = gr.Blocks()
with ui:
	with gr.Tab("Inference"):
		with gr.Row():
			with gr.Column():
				layout["inference"]["inputs"]["text"] = gr.Textbox(lines=4, value="Your prompt here", label="Input Prompt")
		with gr.Row():
			with gr.Column():
				layout["inference"]["inputs"]["reference"] = gr.Audio(label="Audio Input", source="upload", type="filepath")
			with gr.Column():
				layout["inference"]["inputs"]["steps"] = gr.Slider(value=450, minimum=2, maximum=1024, step=1, label="Steps")
				layout["inference"]["inputs"]["ar-temp"] = gr.Slider(value=0.95, minimum=0.0, maximum=1.2, step=0.05, label="Temperature (AR)")
				layout["inference"]["inputs"]["nar-temp"] = gr.Slider(value=0.25, minimum=0.0, maximum=1.2, step=0.05, label="Temperature (NAR)")
			with gr.Column():
				layout["inference"]["buttons"]["start"] = gr.Button(value="Inference")
				# layout["inference"]["stop"] = gr.Button(value="Stop")
				layout["inference"]["outputs"]["output"] = gr.Audio(label="Output")

		layout["inference"]["buttons"]["start"].click(
			fn=do_inference,
			inputs=[ x for x in layout["inference"]["inputs"].values() if x is not None],
			outputs=[ x for x in layout["inference"]["outputs"].values() if x is not None]
		)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--listen", default=None, help="Path for Gradio to listen on")
parser.add_argument("--share", action="store_true")
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

ui.queue(max_size=8)
ui.launch(share=args.share, server_name=args.listen_host, server_port=args.listen_port)