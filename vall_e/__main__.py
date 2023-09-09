import argparse
from pathlib import Path
from .inference import TTS

def path_list(arg):
	return [Path(p) for p in arg.split(";")]

def main():
	parser = argparse.ArgumentParser("VALL-E TTS")
	parser.add_argument("text")
	parser.add_argument("references", type=path_list)
	parser.add_argument("--out-path", type=Path, default=None)

	parser.add_argument("--yaml", type=Path, default=None)
	parser.add_argument("--ar-ckpt", type=Path, default=None)
	parser.add_argument("--nar-ckpt", type=Path, default=None)
	parser.add_argument("--max-ar-steps", type=int, default=6 * 75)
	parser.add_argument("--ar-temp", type=float, default=1.0)
	parser.add_argument("--nar-temp", type=float, default=1.0)

	parser.add_argument("--top-p", type=float, default=1.0)
	parser.add_argument("--top-k", type=int, default=0)
	parser.add_argument("--repetition-penalty", type=float, default=1.0)
	parser.add_argument("--length-penalty", type=float, default=0.0)

	parser.add_argument("--device", default="cuda")
	args = parser.parse_args()

	tts = TTS( config=args.yaml, ar_ckpt=args.ar_ckpt, nar_ckpt=args.nar_ckpt, device=args.device )
	tts.inference( text=args.text, references=args.references, out_path=args.out_path, max_ar_steps=args.max_ar_steps, ar_temp=args.ar_temp, nar_temp=args.nar_temp, top_p=args.top_p, top_k=args.top_k, repetition_penalty=args.repetition_penalty, length_penalty=args.length_penalty )

if __name__ == "__main__":
	main()
