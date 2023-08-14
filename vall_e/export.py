import argparse

import torch

from .data import get_phone_symmap
from .train import load_engines
from .config import cfg

def main():
	parser = argparse.ArgumentParser("Save trained model to path.")
	#parser.add_argument("--yaml", type=Path, default=None)
	args = parser.parse_args()

	engines = load_engines()
	for name, engine in engines.items():
		outpath = cfg.ckpt_dir / name / "fp32.pth"
		torch.save({
			"global_step": engine.global_step,
			"micro_step": engine.micro_step,
			'module': engine.module.to('cpu', dtype=torch.float32).state_dict(),
			#'optimizer': engine.optimizer.state_dict(),
			'symmap': get_phone_symmap(),
		}, outpath)
		print(f"Exported {name} to {outpath}")

if __name__ == "__main__":
	main()