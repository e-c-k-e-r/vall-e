import argparse
from pathlib import Path
from .inference import TTS
from .config import cfg

def path_list(arg):
	return [Path(p) for p in arg.split(";")]

def main():
	parser = argparse.ArgumentParser("VALL-E TTS")
	parser.add_argument("text")
	parser.add_argument("references", type=path_list)
	parser.add_argument("--language", type=str, default="en")
	parser.add_argument("--out-path", type=Path, default=None)

	parser.add_argument("--yaml", type=Path, default=None)

	parser.add_argument("--max-ar-steps", type=int, default=12 * cfg.dataset.frames_per_second)
	parser.add_argument("--max-nar-levels", type=int, default=7)
	parser.add_argument("--max-ar-context", type=int, default=-1)

	parser.add_argument("--ar-temp", type=float, default=1.0)
	parser.add_argument("--nar-temp", type=float, default=0.01)
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

	parser.add_argument("--device", type=str, default=None)
	parser.add_argument("--amp", action="store_true")
	parser.add_argument("--dtype", type=str, default=None)
	args = parser.parse_args()

	tts = TTS( config=args.yaml, device=args.device, dtype=args.dtype, amp=args.amp )
	tts.inference(
		text=args.text,
		references=args.references,
		language=args.language,
		out_path=args.out_path,
		input_prompt_length=args.input_prompt_length,
		max_ar_steps=args.max_ar_steps, max_nar_levels=args.max_nar_levels,
		max_ar_context=args.max_ar_context,
		ar_temp=args.ar_temp, nar_temp=args.nar_temp,
		min_ar_temp=args.min_ar_temp, min_nar_temp=args.min_nar_temp,
		top_p=args.top_p, top_k=args.top_k,
		repetition_penalty=args.repetition_penalty, repetition_penalty_decay=args.repetition_penalty_decay,
		length_penalty=args.length_penalty,
		beam_width=args.beam_width,
		mirostat_tau=args.mirostat_tau, mirostat_eta=args.mirostat_eta
	)

if __name__ == "__main__":
	main()
