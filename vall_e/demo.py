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
import torch

_logger = logging.getLogger(__name__)

from pathlib import Path

from .inference import TTS
from .config import cfg
from .data import create_train_dataloader, create_val_dataloader, get_random_prompt
from .emb.qnt import decode_to_file
from .metrics import wer, sim_o
from .utils import setup_logging, mean
from .utils.io import json_read, json_write

from tqdm import tqdm, trange

def encode(path):
	if path is None or not path.exists():
		return ""
	return "data:audio/wav;base64," + base64.b64encode(open(path, "rb").read()).decode('utf-8')

def safe_inference( tts, out_path, **kwargs ):
	if args.skip_existing and out_path.exists():
		return
	
	try:
		tts.inference( out_path=out_path, **kwargs )
	except Exception as e:
		raise e
		print(f'Error while processing {out_path}: {e}')

def safe_batched_inference( tts, **kwargs ):
	try:
		tts.batched_inference( **kwargs )
	except Exception as e:
		raise e
		print(f'Error while processing batch: {e}')

def process_batch( tts, inputs, kwargs={} ):
	kwargs = kwargs | dict(
		texts=[ x[0] for x in inputs ],
		references=[ x[1] for x in inputs ],
		languages=[ x[2] for x in inputs ],
		out_paths=[ x[3] for x in inputs ],
	)
	safe_batched_inference( tts, **kwargs )

# Would be downright sugoi if I could incorporate this with into __main__
def main():
	parser = argparse.ArgumentParser("VALL-E TTS Demo")

	parser.add_argument("--yaml", type=Path, default=None)
	parser.add_argument("--model", type=Path, default=None)
	parser.add_argument("--batch-size", type=int, default=cfg.inference.batch_size)
	
	parser.add_argument("--demo-dir", type=Path, default=None)
	parser.add_argument("--skip-existing", action="store_true")
	parser.add_argument("--dataset-dir-name", type=str, default="")
	parser.add_argument("--dataset-dir-name-prefix", type=str, default=None)
	parser.add_argument("--sample-from-dataset", action="store_true")
	parser.add_argument("--skip-loading-dataloader", action="store_true")
	parser.add_argument("--dataset-samples", type=int, default=0)
	parser.add_argument("--audio-path-root", type=str, default=None)
	parser.add_argument("--preamble", type=str, default=None)
	parser.add_argument("--output-filename", type=str, default="index.html")
	
	parser.add_argument("--language", type=str, default="auto")
	parser.add_argument("--task", type=str, default="tts")
	parser.add_argument("--modality", type=str, default="auto")
	parser.add_argument("--out-path", type=Path, default=None)

	parser.add_argument("--max-duration", type=int, default=12 * cfg.dataset.frames_per_second)
	parser.add_argument("--max-steps", type=int, default=30)
	parser.add_argument("--max-levels", type=int, default=7)

	parser.add_argument("--ar-temperature", type=float, default=1.0)
	parser.add_argument("--nar-temperature", type=float, default=0.0)
	parser.add_argument("--min-ar-temperature", type=float, default=-1.0)
	parser.add_argument("--min-nar-temperature", type=float, default=-1.0)
	parser.add_argument("--input-prompt-length", type=float, default=5.0)
	parser.add_argument("--input-prompt-prefix", action="store_true")
	parser.add_argument("--prefix-silence", type=float, default=0.0)
	parser.add_argument("--cfg-strength", type=float, default=1.0)
	parser.add_argument("--cfg-rescale", type=float, default=0.75)

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
	
	parser.add_argument("--seed", type=int, default=None)

	parser.add_argument("--device", type=str, default=None)
	parser.add_argument("--amp", action="store_true")
	parser.add_argument("--dtype", type=str, default=None)
	parser.add_argument("--attention", type=str, default="auto")

	parser.add_argument("--random-prompts", action="store_true")
	parser.add_argument("--lora", action="store_true")
	parser.add_argument("--comparison", type=str, default=None)
	
	parser.add_argument("--transcription-model", type=str, default="openai/whisper-large-v3")
	parser.add_argument("--speaker-similarity-model", type=str, default="microsoft/wavlm-large")
	
	args = parser.parse_args()

	config = None
	if args.yaml:
		config = args.yaml
	elif args.model:
		config = args.model
	
	tts = TTS( config=config, lora=args.lora, device=args.device, dtype=args.dtype, amp=args.amp, attention=args.attention )

	if not args.demo_dir:
		args.demo_dir = Path("./data/demo/")

	if not args.preamble:
		args.preamble = "\n".join([
			"Past model demo pages: <a href=\"/models/ar+nar-len-llama-8 (ar+nar).html\">[ar+nar-len-llama-8 (ar+nar)]</a> <a href=\"/models/ar+nar-len-llama-8 (nar-len).html\">[ar+nar-len-llama-8 (nar-len)]</a> <a href=\"/models/ar+nar-llama-8 (ar+nar).html\">[ar+nar-llama-8 (ar+nar)]</a> | Old demo pages: <a href=\"/loras/index.html\">[1]</a> <a href=\"/loras.html\">[2]</a> <a href=\"/old/2024.10.25.html\">[3]</a> <a href=\"/old/2024.12.15.html\">[4]</a> <a href=\"/old/2024.12.16.html\">[5]</a>",
			"<br>",
			"<br>",
			"Below are some samples from my VALL-E implementation: <a href=\"https://git.ecker.tech/mrq/vall-e/\">https://git.ecker.tech/mrq/vall-e/</a>.",
			"<br>",
			"Objective metrics are computed by:",
			"<ul>",
			"	<li>WER/CER: transcribing (openai/whisper-base) then comparing the un-normalized word error rate on the phonemized transcriptions.</li>",
			" 	<li>SIM-O: retrieving the speaker embeddings of the output audio and the input prompt, from a finetune of WavLM for speaker verification (microsoft/wavlm-large), and computing the cosine similarity between the embeddings.</li>",
			"</ul>",
			"Tables marked as \"Validation\" are speakers/samples not seen to the model.",
			"<br>",
			f"These samples were generated using <pre>--ar-temperature={args.ar_temperature} --nar-temperature={args.nar_temperature} --cfg-strength={args.cfg_strength} --max-steps={args.max_steps} --top-k={args.top_k} --dtype={args.dtype}</pre>",
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
	elif args.comparison == "modality":
		comparison_kwargs["suffix"] = "modality"
		comparison_kwargs["titles"] = [f"AR+NAR", f"NAR-len"]

		comparison_kwargs["disabled"]["modality"] = "ar+nar"
		comparison_kwargs["disabled"]["cfg_strength"] = 0.0

		comparison_kwargs["enabled"]["modality"] = "nar-len"
		comparison_kwargs["enabled"]["cfg_strength"] = 3.0
	elif args.comparison == "cfg-strength":
		current_cfg_strength = 3.0
		other_cfg_strength = 0.0

		comparison_kwargs["suffix"] = f"no_cfg_strength"
		comparison_kwargs["titles"] = [f"CFG {current_cfg_strength}", f"CFG {other_cfg_strength}"]

		comparison_kwargs["disabled"]["cfg_strength"] = current_cfg_strength
		comparison_kwargs["enabled"]["cfg_strength"] = other_cfg_strength
	elif args.comparison:
		raise Exception(f"Unrecognized comparison flag: {args.comparison}")

	setup_logging()

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
		input_prompt_length=args.input_prompt_length,
		input_prompt_prefix=args.input_prompt_prefix,
		prefix_silence=args.prefix_silence,
		cfg_strength=args.cfg_strength,
		cfg_rescale=args.cfg_rescale,
		
		seed = args.seed if args.seed else int(time.time()),
		tqdm = True,
		batch_size = args.batch_size,
	)

	# replace values in our template
	html = html.replace(r"${PREAMBLE}", args.preamble )
	html = html.replace(r"${SETTINGS}", str(sampling_kwargs))

	# pull from provided samples
	samples_dirs = {}

	# only add the existing librispeech validation dataset if i'm doing validation so I can stop commenting this out
	if not args.dataset_dir_name:
		samples_dirs["librispeech"] = args.demo_dir / "datasets" / "librispeech"
	else:
		if "validation" in args.dataset_dir_name:
			samples_dirs["librispeech"] = args.demo_dir / "datasets" / "librispeech"

		# automatically pull from anything under the dataset dir
		if args.dataset_dir_name.endswith("/*"):
			args.dataset_dir_name = args.dataset_dir_name[:-2]
			datasets = [ dir for dir in (args.demo_dir / args.dataset_dir_name).iterdir() if dir.is_dir() ]
			for path in datasets:
				samples_dirs[path.name] = path
		# user provided dataset
		elif (args.demo_dir / args.dataset_dir_name).exists():
			samples_dirs["dataset"] = args.demo_dir / args.dataset_dir_name

	# pull from dataset samples
	if args.sample_from_dataset:
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
			index = i if not cfg.dataset.sample_shuffle else random.randint( 0, len( dataloader.dataset ) - 1 )
			batch = dataloader.dataset[index]

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

	inputs = []
	outputs = []
	metrics_inputs = []
	comparison_inputs = []
	for dataset_name, sample_dir in samples_dirs.items():
		if not sample_dir.exists():
			continue
		
		samples = []
		speakers = [ dir for dir in sample_dir.iterdir() if dir.is_dir() ]
		speakers.sort()
		sources = [ "ms_valle", "f5" ] if dataset_name == "librispeech" else []

		# generate demo output
		for dir in tqdm(speakers, desc=f"Preparing demo for {dataset_name}"):
			# bail if too many samples
			if args.dataset_samples and len(samples) >= args.dataset_samples:
				break

			text = open(dir / "prompt.txt", encoding="utf-8").read()
			language = open(dir / "language.txt").read() if (dir / "language.txt").exists() else "en"
			prompt = dir / "prompt.wav"
			reference = dir / "reference.wav"
			metrics_path = dir / "metrics.json"
			out_path = dir / "out" / "ours.wav"
			out_path_comparison = dir / "out" / f"ours_{comparison_kwargs['suffix']}.wav"
			external_sources = [ dir / "out" / f"{source}.wav" for source in sources ]

			audio_samples = [ prompt, out_path ]
			if args.comparison:
				audio_samples += [ out_path_comparison ]
			audio_samples += [ p if p.exists() else None for p in external_sources ]

			"""
			# manual invocation
			cmd = f'python3 -m vall_e --yaml="{args.yaml}" "{reference}" "{text}" --out-path={out_path}'
			# F5
			cmd = f'python inference-cli.py --model "F5-TTS" --ref_audio "{reference}" --gen_text "{text}" --output_dir "{out_path.parent}"'
			"""

			if not args.random_prompts or dataset_name == "librispeech":
				audio_samples += [ reference ]

			samples.append((
				text,
			 	audio_samples,
			))

			# segregate comparisons into its own batch because they use different kwargs (and I do not support variadic-batched kwargs)
			if args.comparison:
				should_generate = (args.skip_existing and not out_path_comparison.exists()) or not (args.skip_existing)
				
				if should_generate:
					comparison_inputs.append((text, prompt, language, out_path_comparison))
				
				metrics_inputs.append((dataset_name, text, language, out_path_comparison, prompt, reference, metrics_path))

			should_generate = (args.skip_existing and not out_path.exists()) or not (args.skip_existing)

			if should_generate:
				inputs.append((text, prompt, language, out_path))
			
			metrics_inputs.append((dataset_name, text, language, out_path, prompt, reference, metrics_path))

		outputs.append((dataset_name, samples))

	if inputs:
		process_batch( tts, inputs, sampling_kwargs | (comparison_kwargs["disabled"] if args.comparison else {}) )
	
	if comparison_inputs:
		process_batch( tts, comparison_inputs, sampling_kwargs | (comparison_kwargs["enabled"] if args.comparison else {}) )

	metrics_map = {}
	for dataset_name, text, language, out_path, prompt_path, reference_path, metrics_path in tqdm(metrics_inputs, desc="Calculating metrics"):
		calculate = not metrics_path.exists() or (metrics_path.stat().st_mtime < out_path.stat().st_mtime)

		if calculate:
			# computes based on word transcriptions outright
			wer_score, cer_score = wer( out_path, text, language=language, device=tts.device, dtype=tts.dtype, model_name=args.transcription_model, phonemize=False )
			# compute on words as well, but does not normalize
			wer_un_score, cer_un_score = wer( out_path, text, language=language, device=tts.device, dtype=tts.dtype, model_name=args.transcription_model, phonemize=False, normalize=False )
			# computes on phonemes instead
			pwer_score, per_score = wer( out_path, text, language=language, device=tts.device, dtype=tts.dtype, model_name=args.transcription_model, phonemize=True )

			sim_o_score = sim_o( out_path, prompt_path, device=tts.device, dtype=tts.dtype, model_name=args.speaker_similarity_model )

			metrics = {"wer": wer_score, "cer": cer_score, "sim-o": sim_o_score, "per": per_score, "pwer": pwer_score, "wer_un": wer_un_score, "cer_un": cer_un_score }
			json_write( metrics, metrics_path )
		else:
			metrics = json_read( metrics_path )
			wer_score, cer_score, per_score, sim_o_score = metrics["wer"], metrics["cer"], metrics["per"], metrics["sim-o"]

		if dataset_name not in metrics_map:
			metrics_map[dataset_name] = {}

		metrics_map[dataset_name][out_path] = (wer_score, cer_score, per_score, sim_o_score)

	# collate entries into HTML
	tables = []
	for dataset_name, samples in outputs:
		table = "\t\t<h3>${DATASET_NAME}</h3>\n\t\t<p><b>Average WER:</b> ${WER}<br><b>Average CER:</b> ${CER}<br><b>Average PER:</b> ${PER}<br><b>Average SIM-O:</b> ${SIM-O}<br></p>\n\t\t<table>\n\t\t\t<thead>\n\t\t\t\t<tr>\n\t\t\t\t\t<th>Text</th>\n\t\t\t\t\t<th>WER↓</th>\n\t\t\t\t\t<th>CER↓</th>\n\t\t\t\t\t<th>SIM-O↑</th>\n\t\t\t\t\t<th>Prompt</th>\n\t\t\t\t\t<th>Our VALL-E</th>\n\t\t\t\t\t<!--th>Original VALL-E</th-->\n\t\t\t\t\t<!--th>F5-TTS</th-->\n\t\t\t\t\t<th>Ground Truth</th>\n\t\t\t\t</tr>\n\t\t\t</thead>\n\t\t\t<tbody>${SAMPLES}</tbody>\n\t\t</table>"
		samples = [
			f'\n\t\t\t<tr>\n\t\t\t\t<td>{text}</td>'+
			"".join([
				f'\n\t\t\t\t<td>{metrics_map[dataset_name][audios[1]][0]:.3f}</td><td>{metrics_map[dataset_name][audios[1]][1]:.3f}</td><td>{metrics_map[dataset_name][audios[1]][2]:.3f}</td><td>{metrics_map[dataset_name][audios[1]][3]:.3f}</td>'
			] ) +
			"".join( [
				f'\n\t\t\t\t<td><audio controls="controls" preload="none"><source src="{str(audio).replace(str(args.demo_dir), args.audio_path_root) if args.audio_path_root else encode(audio)}"/></audio></td>'
				for audio in audios
			] )+
			'\n\t\t\t</tr>'
			for text, audios in samples
		]

		# write audio into template
		table = table.replace("${WER}", f'{mean([ metrics[0] for metrics in metrics_map[dataset_name].values() ]):.3f}' )
		table = table.replace("${CER}", f'{mean([ metrics[1] for metrics in metrics_map[dataset_name].values() ]):.3f}' )
		table = table.replace("${PER}", f'{mean([ metrics[2] for metrics in metrics_map[dataset_name].values() ]):.3f}' )
		table = table.replace("${SIM-O}", f'{mean([ metrics[3] for metrics in metrics_map[dataset_name].values() ]):.3f}' )

		table = table.replace("${DATASET_NAME}", dataset_name)
		table = table.replace("${SAMPLES}", "\n".join( samples ) )
		tables.append( table )
	
	html = html.replace("${TABLES}", "\n".join( tables ))

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
