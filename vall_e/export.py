import argparse

import torch

from .data import get_phone_symmap
from .train import load_engines

def load_models():
	models = {}
	engines = load_engines()
	for name in engines:
		model = engines[name].module.cpu()
		
		model.phone_symmap = get_phone_symmap()

		models[name] = model

	return models

def main():
	parser = argparse.ArgumentParser("Save trained model to path.")
	parser.add_argument("path")
	args = parser.parse_args()

	models = load_models()
	for name in models:
		model = models[name]

		outpath = f'{args.path}/{name}.pt'
		torch.save({
			'module': model.state_dict()
		}, outpath)
		print(f"Exported {name} to {outpath}")

if __name__ == "__main__":
	main()