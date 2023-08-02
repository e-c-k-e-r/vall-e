import argparse
from pathlib import Path
from .inference import TTS

def main():
	parser = argparse.ArgumentParser("VALL-E TTS")
	parser.add_argument("text")
	parser.add_argument("reference", type=Path)
	parser.add_argument("out_path", type=Path)
	parser.add_argument("--yaml", type=Path, default=None)
	parser.add_argument("--ar-ckpt", type=Path, default=None)
	parser.add_argument("--nar-ckpt", type=Path, default=None)
	parser.add_argument("--max-ar-steps", type=int, default=6 * 75)
	parser.add_argument("--ar-temp", type=float, default=1.0)
	parser.add_argument("--nar-temp", type=float, default=1.0)
	parser.add_argument("--device", default="cuda")
	args = parser.parse_args()

	tts = TTS( config=args.yaml, ar_ckpt=args.ar_ckpt, nar_ckpt=args.nar_ckpt, device=args.device )
	tts.inference( text=args.text, reference=args.reference, out_path=args.out_path, max_ar_samples=args.max_ar_samples, ar_temp=args.ar_temp, nar_temp=args.nar_temp )

if __name__ == "__main__":
	main()
