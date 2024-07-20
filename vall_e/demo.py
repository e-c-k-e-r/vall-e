"""
A helper script to generate a demo page.

Layout as expected:
    ./data/demo/:
        {speaker ID}:
            out:
                ours.wav (generated)
                ms_valle.wav
                yourtts.wav
            prompt.txt (text to generate)
            prompt.wav (reference clip to serve as the prompt)
            reference.wav (ground truth utterance)

Will also generate samples from a provided datset, if requested.
"""

import argparse
import base64
import random

from pathlib import Path

from .inference import TTS
from .config import cfg
from .data import create_train_dataloader, create_val_dataloader
from .emb.qnt import decode_to_file

from tqdm import tqdm

def encode(path):
	return "data:audio/wav;base64," + base64.b64encode(open(path, "rb").read()).decode('utf-8')

# Would be downright sugoi if I could incorporate this with into __main__
def main():
	parser = argparse.ArgumentParser("VALL-E TTS Demo")

	parser.add_argument("--yaml", type=Path, default=None)
	
	parser.add_argument("--demo-dir", type=Path, default=None)
	parser.add_argument("--skip-existing", action="store_true")
	parser.add_argument("--sample-from-dataset", action="store_true")
	parser.add_argument("--dataset-samples", type=int, default=0)
	parser.add_argument("--audio-path-root", type=str, default=None)
	
	parser.add_argument("--language", type=str, default="en")

	parser.add_argument("--max-ar-steps", type=int, default=12 * cfg.dataset.frames_per_second)
	parser.add_argument("--max-nar-levels", type=int, default=7)

	parser.add_argument("--ar-temp", type=float, default=1.0)
	parser.add_argument("--nar-temp", type=float, default=0.0)
	parser.add_argument("--min-ar-temp", type=float, default=-1.0)
	parser.add_argument("--min-nar-temp", type=float, default=-1.0)
	parser.add_argument("--input-prompt-length", type=float, default=3.0)

	parser.add_argument("--top-p", type=float, default=1.0)
	parser.add_argument("--top-k", type=int, default=16)
	parser.add_argument("--repetition-penalty", type=float, default=1.0)
	parser.add_argument("--repetition-penalty-decay", type=float, default=0.0)
	parser.add_argument("--length-penalty", type=float, default=0.0)
	parser.add_argument("--beam-width", type=int, default=0)
	
	parser.add_argument("--mirostat-tau", type=float, default=0)
	parser.add_argument("--mirostat-eta", type=float, default=0)
	
	parser.add_argument("--seed", type=int, default=None)

	parser.add_argument("--device", type=str, default=None)
	parser.add_argument("--amp", action="store_true")
	parser.add_argument("--dtype", type=str, default=None)
	
	args = parser.parse_args()
	
	tts = TTS( config=args.yaml, device=args.device, dtype=args.dtype, amp=args.amp )

	if not args.demo_dir:
		args.demo_dir = Path("./data/demo/")

	entries = []

	# pull from provided samples
	sample_dir = args.demo_dir / "librispeech"
	if sample_dir.exists():
		speakers = [ dir for dir in sample_dir.iterdir() if dir.is_dir() ]
		sources = ["ms_valle", "yourtts"]

		# generate demo output
		for dir in tqdm(speakers, desc=f"Generating demo for speaker"):
			text = open(dir / "prompt.txt").read()
			prompt = dir / "prompt.wav"
			out_path = dir / "out" / "ours.wav"

			entries.append((
				text,
			 	[ prompt, dir / "reference.wav", out_path ] + [ dir / "out" / f"{source}.wav" for source in sources ]
			))

			if args.skip_existing and out_path.exists():
				continue

			tts.inference(
				text=text,
				references=[prompt],
				language=args.language,
				out_path=out_path,
				input_prompt_length=args.input_prompt_length,
				max_ar_steps=args.max_ar_steps, max_nar_levels=args.max_nar_levels,
				ar_temp=args.ar_temp, nar_temp=args.nar_temp,
				min_ar_temp=args.min_ar_temp, min_nar_temp=args.min_nar_temp,
				top_p=args.top_p, top_k=args.top_k,
				repetition_penalty=args.repetition_penalty, repetition_penalty_decay=args.repetition_penalty_decay,
				length_penalty=args.length_penalty,
				beam_width=args.beam_width,
				mirostat_tau=args.mirostat_tau, mirostat_eta=args.mirostat_eta,
				seed=args.seed,
				tqdm=False,
			)
		
		entries = [
			f'<tr><td>{text}</td>'+
			"".join( [
				f'<td><audio controls="controls" autobuffer="autobuffer"><source src="{str(audio).replace(str(args.demo_dir), args.audio_path_root) if args.audio_path_root else encode(audio)}"/></audio></td>'
				for audio in audios
			] )+
			'</tr>'
			for text, audios in entries
		]

	# read html template
	html = open(args.demo_dir / "index.template.html", "r", encoding="utf-8").read()
	# create html table, in one messy line
	# replace in our template
	html = html.replace(r"${ENTRIES}", "\n".join(entries) )

	samples = []

	# pull from dataset samples
	if args.sample_from_dataset:
		print("Loading dataloader...")
		dataloader = create_train_dataloader()
		print("Loaded dataloader.")

		num = args.dataset_samples if args.dataset_samples else cfg.evaluation.size

		length = len( dataloader.dataset )
		for i in range( num ):
			idx = random.randint( 0, length )
			batch = dataloader.dataset[idx]

			dir = args.demo_dir / "samples" / f'{i}'

			(dir / "out").mkdir(parents=True, exist_ok=True)

			text = batch["text_string"]
			
			prompt = dir / "prompt.wav"
			reference = dir / "reference.wav"
			out_path = dir / "out" / "ours.wav"

			samples.append((
				text,
			 	[ prompt, reference, out_path ]
			))

			if args.skip_existing and out_path.exists():
				continue

			decode_to_file( batch["proms"].to("cuda"), prompt, device="cuda" )
			decode_to_file( batch["resps"].to("cuda"), reference, device="cuda" )

			tts.inference(
				text=text,
				references=[prompt],
				language=args.language,
				out_path=out_path,
				input_prompt_length=args.input_prompt_length,
				max_ar_steps=args.max_ar_steps, max_nar_levels=args.max_nar_levels,
				ar_temp=args.ar_temp, nar_temp=args.nar_temp,
				min_ar_temp=args.min_ar_temp, min_nar_temp=args.min_nar_temp,
				top_p=args.top_p, top_k=args.top_k,
				repetition_penalty=args.repetition_penalty, repetition_penalty_decay=args.repetition_penalty_decay,
				length_penalty=args.length_penalty,
				beam_width=args.beam_width,
				mirostat_tau=args.mirostat_tau, mirostat_eta=args.mirostat_eta,
				seed=args.seed,
				tqdm=False,
			)

		samples = [
			f'<tr><td>{text}</td>'+
			"".join( [
				f'<td><audio controls="controls" autobuffer="autobuffer"><source src="{str(audio).replace(str(args.demo_dir), args.audio_path_root) if args.audio_path_root else encode(audio)}"/></audio></td>'
				for audio in audios
			] )+
			'</tr>'
			for text, audios in samples
		]

	html = html.replace(r"${SAMPLES}", "\n".join(samples) )

	open( args.demo_dir / "index.html", "w", encoding="utf-8" ).write( html )

if __name__ == "__main__":
	main()
