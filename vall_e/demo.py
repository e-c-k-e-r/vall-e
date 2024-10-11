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
import logging
import time

_logger = logging.getLogger(__name__)

from pathlib import Path

from .inference import TTS
from .config import cfg
from .data import create_train_dataloader, create_val_dataloader, get_random_prompt
from .emb.qnt import decode_to_file

from tqdm import tqdm, trange

def encode(path):
	return "data:audio/wav;base64," + base64.b64encode(open(path, "rb").read()).decode('utf-8')

# Would be downright sugoi if I could incorporate this with into __main__
def main():
	parser = argparse.ArgumentParser("VALL-E TTS Demo")

	parser.add_argument("--yaml", type=Path, default=None)
	
	parser.add_argument("--demo-dir", type=Path, default=None)
	parser.add_argument("--skip-existing", action="store_true")
	parser.add_argument("--dataset-dir-name", type=str, default="dataset")
	parser.add_argument("--sample-from-dataset", action="store_true")
	parser.add_argument("--skip-loading-dataloader", action="store_true")
	parser.add_argument("--dataset-samples", type=int, default=0)
	parser.add_argument("--audio-path-root", type=str, default=None)
	parser.add_argument("--preamble", type=str, default=None)
	
	parser.add_argument("--language", type=str, default="en")

	parser.add_argument("--max-ar-steps", type=int, default=12 * cfg.dataset.frames_per_second)
	parser.add_argument("--max-nar-levels", type=int, default=7)

	parser.add_argument("--ar-temp", type=float, default=1.0)
	parser.add_argument("--nar-temp", type=float, default=0.0)
	parser.add_argument("--min-ar-temp", type=float, default=-1.0)
	parser.add_argument("--min-nar-temp", type=float, default=-1.0)
	parser.add_argument("--input-prompt-length", type=float, default=0.0)

	parser.add_argument("--top-p", type=float, default=1.0)
	parser.add_argument("--top-k", type=int, default=0)
	parser.add_argument("--repetition-penalty", type=float, default=1.0)
	parser.add_argument("--repetition-penalty-decay", type=float, default=0.0)
	parser.add_argument("--length-penalty", type=float, default=0.0)
	parser.add_argument("--beam-width", type=int, default=0)
	
	parser.add_argument("--mirostat-tau", type=float, default=0)
	parser.add_argument("--mirostat-eta", type=float, default=0)

	parser.add_argument("--dry-multiplier", type=float, default=0)
	parser.add_argument("--dry-base", type=float, default=1.75)
	parser.add_argument("--dry-allowed-length", type=int, default=2)
	
	parser.add_argument("--seed", type=int, default=None)

	parser.add_argument("--device", type=str, default=None)
	parser.add_argument("--amp", action="store_true")
	parser.add_argument("--dtype", type=str, default=None)

	parser.add_argument("--random-prompts", action="store_true")
	parser.add_argument("--lora", action="store_true")
	
	args = parser.parse_args()
	
	tts = TTS( config=args.yaml, device=args.device, dtype=args.dtype, amp=args.amp )

	if not args.demo_dir:
		args.demo_dir = Path("./data/demo/")

	if not args.preamble:
		args.preamble = "<br>".join([
			'Below are some samples from my VALL-E implementation: <a href="https://git.ecker.tech/mrq/vall-e/">https://git.ecker.tech/mrq/vall-e/</a>.',
			'Unlike the original VALL-E demo page, I\'m placing emphasis on the input prompt, as the model adheres to it stronger than others.',
		])

	# read html template
	html = open(args.demo_dir / "index.template.html", "r", encoding="utf-8").read()

	# replace values in our template
	html = html.replace(r"${PREAMBLE}", args.preamble )
	html = html.replace(r"${SETTINGS}", str(dict(
		input_prompt_length=args.input_prompt_length,
		max_ar_steps=args.max_ar_steps, max_nar_levels=args.max_nar_levels,
		ar_temp=args.ar_temp, nar_temp=args.nar_temp,
		min_ar_temp=args.min_ar_temp, min_nar_temp=args.min_nar_temp,
		top_p=args.top_p, top_k=args.top_k,
		repetition_penalty=args.repetition_penalty, repetition_penalty_decay=args.repetition_penalty_decay,
		length_penalty=args.length_penalty,
		beam_width=args.beam_width,
		mirostat_tau=args.mirostat_tau, mirostat_eta=args.mirostat_eta,
		dry_multiplier=args.dry_multiplier, dry_base=args.dry_base, dry_allowed_length=args.dry_allowed_length,
	)) )

	# pull from provided samples
	samples_dirs = {
		"librispeech": args.demo_dir / "librispeech",
	}

	if (args.demo_dir / args.dataset_dir_name).exists():
		samples_dirs["dataset"] = args.demo_dir / args.dataset_dir_name

	# pull from dataset samples
	if args.sample_from_dataset:
		cfg.dataset.cache = False
		cfg.dataset.sample_type = "path" if args.lora else "speaker"
		cfg.dataset.tasks_list = [ 'tts' ]

		samples_dirs["dataset"] = args.demo_dir / args.dataset_dir_name

		_logger.info("Loading dataloader...")
		dataloader = create_train_dataloader()
		_logger.info("Loaded dataloader.")

		length = min(len( dataloader.dataset ), cfg.evaluation.batch_size)
		num = args.dataset_samples if args.dataset_samples else length

		for i in trange( num, desc="Sampling dataset for samples" ):
			batch = dataloader.dataset[i]

			dir = args.demo_dir / args.dataset_dir_name / f'{i}'

			(dir / "out").mkdir(parents=True, exist_ok=True)

			metadata = batch["metadata"]

			text = get_random_prompt() if args.random_prompts else metadata["text"]
			language = metadata["language"].lower()
			
			prompt = dir / "prompt.wav"
			reference = dir / "reference.wav"
			out_path = dir / "out" / "ours.wav"

			if args.skip_existing and out_path.exists():
				continue

			open( dir / "prompt.txt", "w", encoding="utf-8" ).write( text )
			open( dir / "language.txt", "w", encoding="utf-8" ).write( language )

			decode_to_file( batch["proms"].to("cuda"), prompt, device="cuda" )
			decode_to_file( batch["resps"].to("cuda"), reference, device="cuda" )

	for k, sample_dir in samples_dirs.items():
		if not sample_dir.exists():
			continue
		
		speakers = [ dir for dir in sample_dir.iterdir() if dir.is_dir() ]
		sources = [ "ms_valle", "yourtts" ]

		samples = []

		# generate demo output
		for dir in tqdm(speakers, desc=f"Generating demo for {k}"):
			text = open(dir / "prompt.txt").read()
			language = open(dir / "language.txt").read() if (dir / "language.txt").exists() else "en"
			prompt = dir / "prompt.wav"
			reference = dir / "reference.wav"
			out_path = dir / "out" / "ours.wav"
			out_path_lora = dir / "out" / "ours_lora.wav"

			extra_sources = [ dir / "out" / f"{source}.wav" for source in sources ] if k == "librispeech" else ([ out_path_lora ] if args.lora else [])

			if not args.random_prompts:
				extra_sources += [ reference ]

			samples.append((
				text,
			 	[ prompt, out_path ] + extra_sources,
			))

			if args.skip_existing and out_path.exists():
				continue

			seed = args.seed if args.seed else int(time.time())

			kwargs = dict(
				text=text,
				references=[prompt],
				language=language,
				input_prompt_length=args.input_prompt_length,
				max_ar_steps=args.max_ar_steps, max_nar_levels=args.max_nar_levels,
				ar_temp=args.ar_temp, nar_temp=args.nar_temp,
				min_ar_temp=args.min_ar_temp, min_nar_temp=args.min_nar_temp,
				top_p=args.top_p, top_k=args.top_k,
				repetition_penalty=args.repetition_penalty, repetition_penalty_decay=args.repetition_penalty_decay,
				length_penalty=args.length_penalty,
				beam_width=args.beam_width,
				mirostat_tau=args.mirostat_tau, mirostat_eta=args.mirostat_eta,
				seed=seed,
				tqdm=False,
			)

			if args.lora:
				tts.enable_lora() # I don't think this is necessary with the below
				kwargs["use_lora"] = True
				try:
					tts.inference( out_path=out_path_lora, **kwargs )
				except Exception as e:
					print(f'Error while processing {out_path}: {e}')
				tts.disable_lora()
				kwargs["use_lora"] = False
			try:
				tts.inference( out_path=out_path, **kwargs )
			except Exception as e:
				print(f'Error while processing {out_path}: {e}')


		# collate entries into HTML
		samples = [
			f'\n\t\t\t<tr>\n\t\t\t\t<td>{text}</td>'+
			"".join( [
				f'\n\t\t\t\t<td><audio controls="controls" preload="none"><source src="{str(audio).replace(str(args.demo_dir), args.audio_path_root) if args.audio_path_root else encode(audio)}"/></audio></td>'
				for audio in audios
			] )+
			'\n\t\t\t</tr>'
			for text, audios in samples
		]

		# write audio into template
		html = html.replace("${"+k.upper()+"_SAMPLES}", "\n".join( samples ) )

		if args.lora:
			if args.random_prompts:
				html = html.replace("<th>Our VALL-E</th>\n\t\t\t\t\t<th>Ground Truth</th>", "<th>Our VALL-E (No LoRA)</th>\n\t\t\t\t\t<th>Our VALL-E (LoRA)</th>")
			else:
				html = html.replace("<th>Our VALL-E</th>", "<th>Our VALL-E (No LoRA)</th>\n\t\t\t\t\t<th>Our VALL-E (LoRA)</th>")

	# write demo page
	open( args.demo_dir / "index.html", "w", encoding="utf-8" ).write( html )

if __name__ == "__main__":
	main()
