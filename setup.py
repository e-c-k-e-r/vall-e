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

platform_dependencies = []

if sys.platform.startswith("win"):
	platform_dependencies += ["psutil"]
else:
	platform_dependencies += ["deepspeed>=0.7.7"]

setup(
	name="vall-e",
	python_requires=">=3.10.0",
	version=write_version("0.0.1"),
	description="An unofficial implementation of the audio LM VALL-E",
	author="ecker",
	author_email="mrq@ecker.tech",
	long_description=long_description,
	long_description_content_type="text/markdown",
	packages=find_packages(),
	install_requires=
		platform_dependencies + [
		# logging niceties
		"coloredlogs>=15.0.1", # barely required
		"humanize>=4.4.0", # not really required
		"matplotlib>=3.6.0", # only required for plotting
		"pandas>=1.5.0", # not really required

		# boiler plate niceties
		#"diskcache>=5.4.0",
		"einops>=0.6.0", # could be replaced
		"tqdm",

		# HF bloat
		"tokenizers",
		"transformers",
		"safetensors",

		# training bloat
		"auraloss[all]", # [all] is needed for MelSTFTLoss
		"h5py",
		"prodigyopt @ git+https://github.com/konstmish/prodigy",

		# practically the reason to use python
		"numpy",
		"torch>=1.13.0",
		"torchaudio>=0.13.0",
		"torchmetrics",

		# core foundations
		"phonemizer>=2.1.0",
		"encodec>=0.1.1",
		"vocos",

		# for the web UI
		"gradio",
		"nltk", # for parsing text inputs down to pieces
		"langdetect", # for detecting the language of a text
		"sounddevice", # for raw playback
	],
	extras_require = {
		"all": [
			# retnet backend (even though two internal copies exist)
			"torchscale @ git+https://git.ecker.tech/mrq/torchscale",
			# bitnet
			"bitnet",
			# mamba
			"causal-conv1d",
			"mamba-ssm",

			#
			"torcheval",

			# attention helpers
			"xformers",
			"sageattention==1.0.6",
			# "flash-attn" --no-build-isolation # commented out right now because I want to query this for Volta freaks like me who can't use it
			
			# other audio backend that doesn't prove fruitful
			"descript-audio-codec",

			# nemo (to-do: cut this down)
			"nemo-toolkit",
			"hydra-core",
			"lightning",
			"sentencepiece"
		]
	},
	url="https://git.ecker.tech/mrq/vall-e",
)
