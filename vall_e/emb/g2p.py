import argparse
import random
import string
import torch

from functools import cache
from pathlib import Path
from phonemizer import phonemize
from phonemizer.backend import BACKENDS

from tqdm import tqdm

try:
	import pykakasi
except Exception as e:
	pass

@cache
def _get_graphs(path):
	with open(path, "r") as f:
		graphs = f.read()
	return graphs

def romanize( runes, sep="" ):	
	kks = pykakasi.kakasi()
	result = kks.convert( runes )
	return sep.join([ res['hira'] for res in result ])

cached_backends = {}
def _get_backend( language="en-us", backend="espeak", punctuation=True, stress=True, strip=True ):
	key = f'{language}_{backend}'
	if key in cached_backends:
		return cached_backends[key]

	if backend == 'espeak':
		phonemizer = BACKENDS[backend]( language, preserve_punctuation=punctuation, with_stress=stress)
	elif backend == 'espeak-mbrola':
		phonemizer = BACKENDS[backend]( language )
	else: 
		phonemizer = BACKENDS[backend]( language, preserve_punctuation=punctuation )

	cached_backends[key] = phonemizer
	return phonemizer


def encode(text: str, language="en-us", backend="auto", punctuation=True, stress=True, strip=True) -> list[str]:
	if language == "en":
		language = "en-us"

	# Convert to kana because espeak does not like kanji...
	if language[:2] == "ja" and backend == "auto":
		text = romanize( text )

	if not backend or backend == "auto":
		backend = "espeak" # if language[:2] != "en" else "festival"

	backend = _get_backend(language=language, backend=backend, stress=stress, strip=strip, punctuation=punctuation)
	if backend is not None:
		tokens = backend.phonemize( [ text ], strip=strip )
	else:
		tokens = phonemize( [ text ], language=language, strip=strip, preserve_punctuation=punctuation, with_stress=stress )
	
	if not len(tokens):
		raise Exception(f"Failed to phonemize, received empty string: {text}")

	return tokens[0]

# Helper function to debug phonemizer
if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("string", type=str)
	parser.add_argument("--language", type=str, default="en-us")
	parser.add_argument("--backend", type=str, default="auto")
	parser.add_argument("--no-punctuation", action="store_true")
	parser.add_argument("--no-stress", action="store_true")
	parser.add_argument("--no-strip", action="store_true")

	args = parser.parse_args()

	phonemes = encode( args.string, language=args.language, backend=args.backend, punctuation=not args.no_punctuation, stress=not args.no_stress, strip=not args.no_strip )
	print( phonemes )