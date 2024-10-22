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
	parser.add_argument("--language", type=str, default="en")
	parser.add_argument("--task", type=str, default="tts")
	parser.add_argument("--out-path", type=Path, default=None)

	parser.add_argument("--yaml", type=Path, default=None)

	parser.add_argument("--max-ar-steps", type=int, default=12 * cfg.dataset.frames_per_second)
	parser.add_argument("--max-nar-levels", type=int, default=7)

	parser.add_argument("--ar-temp", type=float, default=0.0)
	parser.add_argument("--nar-temp", type=float, default=0.0)
	parser.add_argument("--min-ar-temp", type=float, default=-1.0)
	parser.add_argument("--min-nar-temp", type=float, default=-1.0)
	parser.add_argument("--input-prompt-length", type=float, default=3.0)
	parser.add_argument("--input-prompt-prefix", action="store_true")

	parser.add_argument("--top-p", type=float, default=1.0)
	parser.add_argument("--top-k", type=int, default=0)
	parser.add_argument("--min-p", type=float, default=0.0)
	parser.add_argument("--repetition-penalty", type=float, default=1.125)
	parser.add_argument("--repetition-penalty-decay", type=float, default=0.0)
	parser.add_argument("--length-penalty", type=float, default=0.0)
	parser.add_argument("--beam-width", type=int, default=0)
	
	parser.add_argument("--mirostat-tau", type=float, default=0)
	parser.add_argument("--mirostat-eta", type=float, default=0)
	
	parser.add_argument("--dry-multiplier", type=float, default=0)
	parser.add_argument("--dry-base", type=float, default=1.75)
	parser.add_argument("--dry-allowed-length", type=int, default=2)
	
	parser.add_argument("--entropix-sampling", action="store_true")
	
	parser.add_argument("--seed", type=int, default=None)

	parser.add_argument("--device", type=str, default=None)
	parser.add_argument("--amp", action="store_true")
	parser.add_argument("--dtype", type=str, default=None)
	parser.add_argument("--attention", type=str, default=None)
	args = parser.parse_args()

	tts = TTS( config=args.yaml, device=args.device, dtype=args.dtype, amp=args.amp, attention=args.attention )
	output = tts.inference(
		text=args.text,
		references=args.references,
		language=args.language,
		task=args.task,
		out_path=args.out_path,
		input_prompt_length=args.input_prompt_length,
		input_prompt_prefix=args.input_prompt_prefix,
		max_ar_steps=args.max_ar_steps, max_nar_levels=args.max_nar_levels,
		ar_temp=args.ar_temp, nar_temp=args.nar_temp,
		min_ar_temp=args.min_ar_temp, min_nar_temp=args.min_nar_temp,
		top_p=args.top_p, top_k=args.top_k, min_p=args.min_p,
		repetition_penalty=args.repetition_penalty, repetition_penalty_decay=args.repetition_penalty_decay,
		length_penalty=args.length_penalty,
		beam_width=args.beam_width,
		mirostat_tau=args.mirostat_tau, mirostat_eta=args.mirostat_eta,
		dry_multiplier=args.dry_multiplier, dry_base=args.dry_base, dry_allowed_length=args.dry_allowed_length,
		entropix_sampling=args.entropix_sampling,
		seed=args.seed,
	)
	
	if isinstance( output, str ):
		print( output )

if __name__ == "__main__":
	main()
