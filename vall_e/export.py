import argparse

import torch

from .data import get_phone_symmap
from .train import load_engines
from .config import cfg

def main():
	parser = argparse.ArgumentParser("Save trained model to path.")
	parser.add_argument("--module-only", action='store_true')
	args = parser.parse_args()

	if args.module_only:
		cfg.trainer.load_module_only = True

	engines = load_engines()
	engines.export(userdata={"symmap": get_phone_symmap()})

if __name__ == "__main__":
	main()