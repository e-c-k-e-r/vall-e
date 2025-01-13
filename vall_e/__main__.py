import argparse
from pathlib import Path
from .inference import TTS
from .config import cfg

def path_list(arg):
	if not arg:
		return None
	return [Path(p) for p in arg.split(";")]

def main():
	parser = argparse.ArgumentParser("VALL-E TTS")
	parser.add_argument("text")
	parser.add_argument("references", type=path_list, default=None)
	parser.add_argument("--language", type=str, default="auto")
	parser.add_argument("--text-language", type=str, default=None)
	parser.add_argument("--task", type=str, default="tts")
	parser.add_argument("--modality", type=str, default="auto")
	parser.add_argument("--out-path", type=Path, default=None)

	parser.add_argument("--split-text-by", type=str, default="\n")
	parser.add_argument("--context-history", type=int, default=0)
	parser.add_argument("--no-phonemize", action='store_true')

	parser.add_argument("--yaml", type=Path, default=None)
	parser.add_argument("--model", type=Path, default=None)
	parser.add_argument("--lora", type=Path, default=None)

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
	parser.add_argument("--cfg-strength", type=float, default=0.0)
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
	parser.add_argument("--attention", type=str, default=None)
	parser.add_argument("--play", action="store_true")
	args = parser.parse_args()

	config = None

	if args.yaml:
		config = args.yaml
	elif args.model:
		config = args.model

	tts = TTS( config=config, lora=args.lora, device=args.device, dtype=args.dtype, amp=args.amp, attention=args.attention )

	sampling_kwargs = dict(
		split_text_by=args.split_text_by,
		context_history=args.context_history,
		phonemize=not args.no_phonemize,
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
		cfg_rescale=args.cfg_rescale,
	)

	output = tts.inference(
		text=args.text,
		references=args.references,
		text_language=args.text_language,
		language=args.language,
		task=args.task,
		modality=args.modality,
		out_path=args.out_path,
		play=args.play,

		input_prompt_length=args.input_prompt_length,
		load_from_artifact=args.load_from_artifact,

		sampling_kwargs=sampling_kwargs,

		seed=args.seed,
	)
	
	if isinstance( output, str ):
		print( output )

if __name__ == "__main__":
	main()
