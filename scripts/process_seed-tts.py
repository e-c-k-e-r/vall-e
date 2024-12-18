"""
Handles processing seed-tts-eval's dataset into something to be used for vall_e.demo

Reads from meta.lst, a text file where each utterance is formatted as:

<reference path>|<reference text>|<prompt path>|<prompt text>
"""

import os
import json
import argparse
import torch
import shutil
import torchaudio
import numpy as np

from tqdm.auto import tqdm
from pathlib import Path

def process(
	input_dir=Path("./seedtts_testset/en/"),
	list_name="./meta.lst",
	wav_dir="./wavs/",
	output_dir=Path("./dataset/seed-tts-eval-en/"),
):
	language = "auto"
	
	if "en" in str(input_dir):
		language = "en"
	elif "zh" in str(input_dir):
		language = "zh"

	output_dir.mkdir(parents=True, exist_ok=True)

	# read manifest
	lines = open(input_dir / list_name).read()
	lines = lines.split("\n")
	# split it even further
	for line in lines:
		if not line:
			continue
		speaker, text, prompt_wav, prompt_transcription = line.split("|")

		(output_dir / speaker).mkdir(parents=True, exist_ok=True)

		open( output_dir / speaker / "prompt.txt", "w", encoding="utf-8" ).write( text )
		open( output_dir / speaker / "language.txt", "w", encoding="utf-8" ).write( language )

		shutil.copy((input_dir / wav_dir / speaker).with_suffix(".wav"), output_dir / speaker / "reference.wav" )
		shutil.copy(input_dir / prompt_wav, output_dir / speaker / "prompt.wav" )

if __name__ == "__main__":
	process()