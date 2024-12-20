"""
Handles processing NScripter's 0.u file to clean up the pile of audio clips it has

* to-do: also grab transcriptions
"""

import os
import re
import json
import argparse
import torch
import shutil
import torchaudio
import numpy as np

from tqdm.auto import tqdm
from pathlib import Path

def process(
	input_file=Path("./assets/0.u"),
	wav_dir=Path("./arc/"),
	output_dir=Path("./dataset/"),
):
	file = open(input_file, encoding='utf-8').read()

	names = {}
	aliases = {}
	lines = file.split('\n')

	for line in lines:
		if not line.startswith('stralias'):
			continue
		# ick
		try:
			key, path = re.findall(r'^stralias (.+?),"(.+?)"$', line)[0]
			name = key.split("_")[0]
			if name not in names:
				(output_dir / name).mkdir(parents=True, exist_ok=True)
				names[name] = True

			aliases[key] = Path(path)
		except Exception as e:
			pass

	for k, v in aliases.items():
		name = k.split("_")[0]


	print(aliases)

if __name__ == "__main__":
	process()