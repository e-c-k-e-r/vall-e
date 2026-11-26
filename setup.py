import subprocess
import sys
from pathlib import Path
from datetime import datetime
from setuptools import setup, find_packages

def shell(*args):
	out = subprocess.check_output(args)
	return out.decode("ascii").strip()

def write_version(version_core, pre_release=True):
	if pre_release:
		time = shell("git", "log", "-1", "--format=%cd", "--date=iso")
		time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S %z")
		time = time.strftime("%Y%m%d%H%M%S")
		version = f"{version_core}-dev{time}"
	else:
		version = version_core

	with open(Path("vall_e", "version.py"), "w") as f:
		f.write('__version__ = "{}"\n'.format(version))

	return version

with open("README.md", "r") as f:
	long_description = f.read()

setup(
	name="vall-e",
	python_requires=">=3.10.0",
	version=write_version("0.3.0"),
	description="Text-to-speech over NeMo 44.1 kHz FSQ codes with a flow-matching head (VALL-E v3)",
	author="ecker",
	author_email="mrq@ecker.tech",
	long_description=long_description,
	long_description_content_type="text/markdown",
	packages=find_packages(include=["vall_e", "vall_e.*"]),
	package_data={"vall_e": ["harvard_sentences.txt"]},
	install_requires=[
		"torch>=2.4",
		"torchaudio",
		"numpy",
		"tqdm",
		"requests",
		# the codec (vall_e/emb/codec.py: NVIDIA's 44.1 kHz codec in plain PyTorch, read from its .nemo release)
		"pyyaml",
		"huggingface_hub",
		"librosa",  # prosody features (vall_e/emb/prosody.py)
		"soundfile",  # audio files (vall_e/audio.py; torchaudio 2.9+ reads them only through torchcodec and FFmpeg)
		# text
		"phonemizer>=2.1.0",  # needs espeak-ng installed
		"langdetect",
		# scoring: Whisper (WER / CER) and WavLM speaker similarity (fetched through torch.hub's s3prl)
		"transformers",
	],
	extras_require={
		"ja": ["pykakasi"],  # kanji -> hiragana before phonemizing
		"eval": ["inflect"],  # number expansion when normalising English for WER
		"webui": ["gradio"],
	},
	url="https://git.ecker.tech/mrq/vall-e",
)
