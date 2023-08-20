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
	engines.export(userdata={"symmap": get_phone_symmap()})

if __name__ == "__main__":
	main()