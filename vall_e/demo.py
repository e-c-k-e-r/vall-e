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
	if path is None or not path.exists():
		return ""
	return "data:audio/wav;base64," + base64.b64encode(open(path, "rb").read()).decode('utf-8')

# Would be downright sugoi if I could incorporate this with into __main__
def main():
	parser = argparse.ArgumentParser("VALL-E TTS Demo")

	parser.add_argument("--yaml", type=Path, default=None)
	parser.add_argument("--model", type=Path, default=None)
	
	parser.add_argument("--demo-dir", type=Path, default=None)
	parser.add_argument("--skip-existing", action="store_true")
	parser.add_argument("--dataset-dir-name", type=str, default="dataset")
	parser.add_argument("--dataset-dir-name-prefix", type=str, default=None)
	parser.add_argument("--sample-from-dataset", action="store_true")
	parser.add_argument("--skip-loading-dataloader", action="store_true")
	parser.add_argument("--dataset-samples", type=int, default=0)
	parser.add_argument("--audio-path-root", type=str, default=None)
	parser.add_argument("--preamble", type=str, default=None)
	parser.add_argument("--output-filename", type=str, default="index.html")
	
	parser.add_argument("--language", type=str, default="en")
	parser.add_argument("--task", type=str, default="tts")
	parser.add_argument("--modality", type=str, default="auto")
	parser.add_argument("--out-path", type=Path, default=None)

	parser.add_argument("--max-duration", type=int, default=12 * cfg.dataset.frames_per_second)
	parser.add_argument("--max-steps", type=int, default=25)
	parser.add_argument("--max-levels", type=int, default=7)

	parser.add_argument("--ar-temperature", type=float, default=1.0)
	parser.add_argument("--nar-temperature", type=float, default=0.0)
	parser.add_argument("--min-ar-temperature", type=float, default=-1.0)
	parser.add_argument("--min-nar-temperature", type=float, default=-1.0)
	parser.add_argument("--input-prompt-length", type=float, default=3.0)
	parser.add_argument("--input-prompt-prefix", action="store_true")
	parser.add_argument("--prefix-silence", type=float, default=0.0)
	parser.add_argument("--cfg-strength", type=float, default=3.0)

	parser.add_argument("--top-p", type=float, default=1.0)
	parser.add_argument("--top-k", type=int, default=0)
	parser.add_argument("--top-no", type=float, default=0.0)
	parser.add_argument("--min-p", type=float, default=0.0)
	parser.add_argument("--repetition-penalty", type=float, default=1.0)
	parser.add_argument("--repetition-penalty-decay", type=float, default=0.0)
	parser.add_argument("--length-penalty", type=float, default=0.0)
	parser.add_argument("--beam-width", type=int, default=0)
	
	parser.add_argument("--mirostat-tau", type=float, default=0)
	parser.add_argument("--mirostat-eta", type=float, default=0)
	
	parser.add_argument("--dry-multiplier", type=float, default=0)
	parser.add_argument("--dry-base", type=float, default=1.75)
	parser.add_argument("--dry-allowed-length", type=int, default=2)
	
	parser.add_argument("--entropix-sampling", action="store_true")
	
	parser.add_argument("--layer-skip", action="store_true")
	parser.add_argument("--layer-skip-exit-layer", type=int, default=None)
	parser.add_argument("--layer-skip-entropy-threshold", type=int, default=0.1)
	parser.add_argument("--layer-skip-varentropy-threshold", type=int, default=0.1)
	parser.add_argument("--refine-on-stop", action="store_true")

	# experimental settings
	parser.add_argument("--load-from-artifact", type=Path, default=None)
	parser.add_argument("--denoise-start", type=float, default=0.0)
	
	parser.add_argument("--seed", type=int, default=None)

	parser.add_argument("--device", type=str, default=None)
	parser.add_argument("--amp", action="store_true")
	parser.add_argument("--dtype", type=str, default=None)

	parser.add_argument("--random-prompts", action="store_true")
	parser.add_argument("--lora", action="store_true")
	parser.add_argument("--comparison", type=str, default=None)
	
	args = parser.parse_args()

	config = None
	if args.yaml:
		config = args.yaml
	elif args.model:
		config = args.model
	
	tts = TTS( config=config, lora=args.lora, device=args.device, dtype=args.dtype, amp=args.amp )

	if not args.demo_dir:
		args.demo_dir = Path("./data/demo/")

	if not args.preamble:
		args.preamble = "<br>".join([
			'Below are some samples from my VALL-E implementation: <a href="https://git.ecker.tech/mrq/vall-e/">https://git.ecker.tech/mrq/vall-e/</a>.',
			'Unlike the original VALL-E demo page, I\'m placing emphasis on the input prompt, as the model adheres to it stronger than others.',
		])

	# comparison kwargs
	comparison_kwargs = {
		"titles": [],
		"suffix": "diff",
		"enabled": {},
		"disabled": {}
	}

	if args.lora:
		args.comparison = "lora"

	# to-do: just make this mappable
	if args.comparison == "lora":
		comparison_kwargs["suffix"] = "no_lora"
		comparison_kwargs["titles"] = ["LoRA", "No LoRA"]

		comparison_kwargs["disabled"]["use_lora"] = True
		comparison_kwargs["disabled"]["ar_temperature"] = 0.0
		comparison_kwargs["enabled"]["use_lora"] = False
		comparison_kwargs["enabled"]["ar_temperature"] = 0.95
	elif args.comparison == "entropix-sampling":
		comparison_kwargs["suffix"] = "entropix_sampling"
		comparison_kwargs["titles"] = ["Without Entropix", "With Entropix"]	

		comparison_kwargs["disabled"]["entropix_sampling"] = False
		comparison_kwargs["disabled"]["ar_temperature"] = args.ar_temperature
		comparison_kwargs["disabled"]["top_k"] = args.top_k
		comparison_kwargs["disabled"]["top_p"] = args.top_p
		comparison_kwargs["enabled"]["entropix_sampling"] = True
		comparison_kwargs["enabled"]["ar_temperature"] = 0.666
		comparison_kwargs["enabled"]["top_k"] = 27
		comparison_kwargs["enabled"]["top_p"] = 0.9
	elif args.comparison == "layerskip":
		comparison_kwargs["suffix"] = "layerskip"
		comparison_kwargs["titles"] = [f"Without LayerSkip", "With LayerSkip"]

		comparison_kwargs["disabled"]["layer_skip"] = False
		comparison_kwargs["enabled"]["layer_skip"] = True
	elif args.comparison == "refine-on-stop":
		comparison_kwargs["suffix"] = "refine-on-stop"
		comparison_kwargs["titles"] = [f"Without Ro<S>", "With Ro<S>"]

		comparison_kwargs["disabled"]["refine_on_stop"] = False
		comparison_kwargs["enabled"]["refine_on_stop"] = True
	elif args.comparison == "ar-temp":
		current_temperature = args.ar_temperature
		other_temperature = 1.0

		comparison_kwargs["suffix"] = "temperature"
		comparison_kwargs["titles"] = [f"Temp: {current_temperature:.2f}", f"Temp: {other_temperature:.2f}"]

		comparison_kwargs["disabled"]["ar_temperature"] = current_temperature
		comparison_kwargs["enabled"]["ar_temperature"] = other_temperature
	elif args.comparison == "input-prompt-length":
		current_length = args.input_prompt_length
		other_length = 3.0

		comparison_kwargs["suffix"] = "input_prompt_length"
		comparison_kwargs["titles"] = [f"Prompt Length: {current_length:.2f}s", f"Prompt Length: {other_length:.2f}s"]

		comparison_kwargs["disabled"]["input_prompt_length"] = current_length
		comparison_kwargs["enabled"]["input_prompt_length"] = other_length
	elif args.comparison == "dtype":
		current_dtype = cfg.inference.weight_dtype
		other_dtype = "float32"

		if current_dtype == "float16":
			other_dtype = "bfloat16"
		elif current_dtype == "bfloat16":
			other_dtype = "float16"

		comparison_kwargs["suffix"] = f"dtype_{other_dtype}"
		comparison_kwargs["titles"] = [f"With {current_dtype}", f"With {other_dtype}"]

		comparison_kwargs["disabled"]["dtype"] = current_dtype
		comparison_kwargs["enabled"]["dtype"] = other_dtype
	elif args.comparison == "amp":
		current_amp = cfg.inference.weight_amp
		other_amp = not current_amp

		comparison_kwargs["suffix"] = f"with{'out' if not other_amp else ''}_amp"
		comparison_kwargs["titles"] = [f"With {current_amp}", f"With {other_amp}"]

		comparison_kwargs["disabled"]["amp"] = current_amp
		comparison_kwargs["enabled"]["amp"] = other_amp
	elif args.comparison == "cfg-strength":
		current_cfg_strength = 3.0
		other_cfg_strength = 0.0

		comparison_kwargs["suffix"] = f"no_cfg_strength"
		comparison_kwargs["titles"] = [f"CFG {current_cfg_strength}", f"CFG {other_cfg_strength}"]

		comparison_kwargs["disabled"]["cfg_strength"] = current_cfg_strength
		comparison_kwargs["enabled"]["cfg_strength"] = other_cfg_strength
	elif args.comparison:
		raise Exception(f"Unrecognized comparison flag: {args.comparison}")

	# read html template
	html = open(args.demo_dir / "index.template.html", "r", encoding="utf-8").read()

	sampling_kwargs = dict(
		task=args.task,
		modality=args.modality,
		max_steps=args.max_steps,
		max_levels=args.max_levels,
		max_duration=args.max_duration,
		ar_temperature=args.ar_temperature, nar_temperature=args.nar_temperature,
		min_ar_temperature=args.min_ar_temperature, min_nar_temperature=args.min_nar_temperature,
		top_p=args.top_p, top_k=args.top_k, top_no=args.top_no,min_p=args.min_p,
		repetition_penalty=args.repetition_penalty, repetition_penalty_decay=args.repetition_penalty_decay,
		length_penalty=args.length_penalty,
		beam_width=args.beam_width,
		mirostat_tau=args.mirostat_tau, mirostat_eta=args.mirostat_eta,
		dry_multiplier=args.dry_multiplier, dry_base=args.dry_base, dry_allowed_length=args.dry_allowed_length,
		entropix_sampling=args.entropix_sampling,
		layer_skip=args.layer_skip,
		layer_skip_exit_layer=args.layer_skip_exit_layer,
		layer_skip_entropy_threshold=args.layer_skip_entropy_threshold,
		layer_skip_varentropy_threshold=args.layer_skip_varentropy_threshold,
		refine_on_stop=args.refine_on_stop,
		denoise_start=args.denoise_start,
		input_prompt_length=args.input_prompt_length,
		input_prompt_prefix=args.input_prompt_prefix,
		prefix_silence=args.prefix_silence,
		cfg_strength=args.cfg_strength,
	)

	# replace values in our template
	html = html.replace(r"${PREAMBLE}", args.preamble )
	html = html.replace(r"${SETTINGS}", str(sampling_kwargs))

	# pull from provided samples
	samples_dirs = {
		"librispeech": args.demo_dir / "librispeech",
	}

	if (args.demo_dir / args.dataset_dir_name).exists():
		samples_dirs["dataset"] = args.demo_dir / args.dataset_dir_name

	# pull from dataset samples
	if args.sample_from_dataset:
		cfg.dataset.cache = False
		cfg.dataset.sample_type = "path" if len(cfg.dataset.training) < cfg.evaluation.batch_size else "speaker"
		cfg.dataset.sample_order = "random"
		cfg.dataset.tasks_list = [ 'tts' ]

		samples_dirs["dataset"] = args.demo_dir / args.dataset_dir_name

		_logger.info("Loading dataloader...")
		dataloader = create_train_dataloader()
		_logger.info("Loaded dataloader.")

		length = min(len( dataloader.dataset ), cfg.evaluation.batch_size)
		num = args.dataset_samples if args.dataset_samples else length

		for i in trange( num, desc="Sampling dataset for samples" ):
			index = i if not cfg.dataset.sample_shuffle else random.randint( 0, len( dataloader.dataset ) )
			batch = dataloader.dataset[i]

			if args.dataset_dir_name_prefix:
				dir = args.demo_dir / args.dataset_dir_name / f'{args.dataset_dir_name_prefix}_{i}'
			else:
				dir = args.demo_dir / args.dataset_dir_name / f'{i}'

			(dir / "out").mkdir(parents=True, exist_ok=True)

			metadata = batch["metadata"]

			text = get_random_prompt() if args.random_prompts else metadata["text"]
			#text = get_random_prompt() if i >= (num // 2) else metadata["text"]
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
		
		samples = []
		speakers = [ dir for dir in sample_dir.iterdir() if dir.is_dir() ]
		sources = [ "ms_valle", "f5" ]

		# generate demo output
		for dir in tqdm(speakers, desc=f"Generating demo for {k}"):
			text = open(dir / "prompt.txt", encoding="utf-8").read()
			language = open(dir / "language.txt").read() if (dir / "language.txt").exists() else "en"
			prompt = dir / "prompt.wav"
			reference = dir / "reference.wav"
			out_path = dir / "out" / "ours.wav"
			out_path_comparison = dir / "out" / f"ours_{comparison_kwargs['suffix']}.wav"
			external_sources = [ dir / "out" / f"{source}.wav" for source in sources ]

			audio_samples = [ prompt, out_path ]
			if args.comparison:
				audio_samples += [ out_path_comparison ]
			audio_samples += [ p if p.exists() else None for p in external_sources ]

			if not args.random_prompts or k == "librispeech":
				audio_samples += [ reference ]

			samples.append((
				text,
			 	audio_samples,
			))

			seed = args.seed if args.seed else int(time.time())

			"""
			# manual invocation
			cmd = f'python3 -m vall_e --yaml="{args.yaml}" "{reference}" "{text}" --out-path={out_path}'
			# F5
			cmd = f'python inference-cli.py --model "F5-TTS" --ref_audio "{reference}" --gen_text "{text}" --output_dir "{out_path.parent}"'
			"""

			kwargs = dict(
				text=text,
				references=[prompt],
				language=language,
				seed=seed,
				tqdm=False,
				**sampling_kwargs,
			)

			def safe_inference( out_path=out_path ):
				if args.skip_existing and out_path.exists():
					return
				
				# swap model config swap
				"""
				if "dtype" in kwargs or "amp" in kwargs:
					dtype = kwargs.pop("dtype", args.dtype)
					amp = kwargs.pop("amp", args.amp)
					
					del tts
					tts = TTS( config=args.yaml, device=args.device, dtype=dtype, amp=amp )
				"""
				try:
					tts.inference( out_path=out_path, **kwargs )
				except Exception as e:
					raise e
					print(f'Error while processing {out_path}: {e}')

			if args.comparison:
				kwargs.update( comparison_kwargs["enabled"] )
				safe_inference(out_path_comparison)
				kwargs.update( comparison_kwargs["disabled"] )

			safe_inference()

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

		if args.comparison:
			disabled, enabled = comparison_kwargs["titles"]
			if args.random_prompts:
				html = html.replace("<th>Our VALL-E</th>\n\t\t\t\t\t<th>Ground Truth</th>", f"<th>Our VALL-E ({disabled})</th>\n\t\t\t\t\t<th>Our VALL-E ({enabled})</th>")
			else:
				html = html.replace("<th>Our VALL-E</th>", f"<th>Our VALL-E ({disabled})</th>\n\t\t\t\t\t<th>Our VALL-E ({enabled})</th>")

	# write demo page
	open( args.demo_dir / args.output_filename, "w", encoding="utf-8" ).write( html )

if __name__ == "__main__":
	main()
