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
	pykakasi = None
	print(f'Error while importing pykakasi: {str(e)}')
	pass

try:
	import langdetect
except Exception as e:
	langdetect = None
	print(f'Error while importing langdetect: {str(e)}')

@cache
def detect_language( text ):
	if langdetect is None:
		raise Exception('langdetect is not installed.')
	return langdetect.detect( text )

def _get_graphs(path):
	with open(path, "r") as f:
		graphs = f.read()
	return graphs

@cache
def coerce_to_hiragana( runes, sep="" ):	
	if pykakasi is None:
		raise Exception('pykakasi is not installed.')

	kks = pykakasi.kakasi()
	result = kks.convert( runes )
	return sep.join([ res['hira'] for res in result ])

def coerce_language( lang ):
	# bottle of water vs bo'oh'o'wa'er
	if lang == "en":
		lang = "en-us"
	# quebec probably
	if lang == "fr":
		return "fr-fr"
	# phonemizer/espeak used to have zh refer to mandarin, but was renamed to cmn
	# cmn outputs cringe, but not cmn-latn-pinyin
	if lang == "zh":
		return "cmn-latn-pinyin"
	"""
	things to consider in the future
	en-uk or en-gb
	es-la vs es-es
	pt-br vs pt-pt
	"""
	return lang

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


def encode(text: str, language="auto", backend="auto", punctuation=True, stress=True, strip=True) -> list[str]:
	if language == "auto":
		language = detect_language( text )

	language = coerce_language( language )

	#
	if backend == "auto":
		# Convert to hiragana, as espeak does not like kanji
		if language[:2] == "ja":
			text = coerce_to_hiragana( text )

		# "zh" => "cmn-latn-pinyin"
		elif language == "zh":
			language = "cmn-latn-pinyin"


	if not backend or backend == "auto":
		backend = "espeak" # if language[:2] != "en" else "festival"

	backend = _get_backend(language=language, backend=backend, stress=stress, strip=strip, punctuation=punctuation)
	if backend is not None:
		phonemes = backend.phonemize( [ text ], strip=strip )
	else:
		phonemes = phonemize( [ text ], language=language, strip=strip, preserve_punctuation=punctuation, with_stress=stress )
	
	if not len(phonemes):
		raise Exception(f"Failed to phonemize, received empty string: {text}")

	phonemes = phonemes[0]

	# remap tones
	# technically they can be kept in place and just update the tokenizer, but this would be a bit confusing
	if language == "cmn-latn-pinyin":
		tones = {
			"1": "ˇ",
			"2": "ˉ",
			"3": "ˊ",
			"4": "ˋ",
			"5": "_",
		}
		for k, v in tones.items():
			phonemes = phonemes.replace(k, v)

	return phonemes

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