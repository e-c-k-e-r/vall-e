import argparse
import random
import re
import string
import torch

from functools import cache
from pathlib import Path
# espeak-ng 1.52's phonemizing and phonemizer 3.3's espeak backend, standalone (no espeak-ng install): the same output
from propan2ol import EspeakBackend

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

def detect_language( text ):
	if not text:
		return "en" # to-do: map to a null language

	if langdetect is None:
		raise Exception('langdetect is not installed.')
	return langdetect.detect( text )

def _get_graphs(path):
	with open(path, "r") as f:
		graphs = f.read()
	return graphs

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
	# also just coerces any of the dialects into this (to avoid crimes)
	if lang[:2] == "zh":
		return "cmn-latn-pinyin"
	"""
	things to consider in the future
	en-uk or en-gb
	es-la vs es-es
	pt-br vs pt-pt
	"""
	return lang

# English abbreviations spelled out before espeak: it reads several letter by letter ("Ms." as "em ess") and keeps
# every abbreviation's period as a sentence end ("Mr. Smith" -> "mˈɪstɚ. smˈɪθ")
ABBREVIATIONS = dict(Mr="Mister", Mrs="Missus", Ms="Miz", Dr="Doctor", Jr="Junior", Sr="Senior", Lt="Lieutenant", Sgt="Sergeant",
	Capt="Captain", Prof="Professor", Mt="Mount", vs="versus", etc="et cetera")
_ABBREV = re.compile(r"\b(" + "|".join(ABBREVIATIONS) + r"|St)\.(\s+(?=\S)|$)")

def normalize_en( text ):
	def sub( m ):
		word, gap = m.group(1), m.group(2)
		nxt = text[m.end():m.end() + 1]
		if word == "St":
			# "Baker St." / "Main St. Then": after a name it's a street; "on St. Patrick" / "St. Louis": a saint
			prev = text[:m.start()].rstrip().split(" ")[-1:]
			word = "Street" if (prev and prev[0][:1].isupper()) or not nxt.isupper() else "Saint"
		else:
			word = ABBREVIATIONS[word]
		# the period stays only where the sentence ends: the end of the text, or "etc." / "St." before a capital
		end = not nxt or (m.group(1) in ("etc", "St") and word != "Saint" and nxt.isupper())
		return word + ("." if end else "") + (gap if gap else "")
	return _ABBREV.sub( sub, text )

cached_backends = {}
def _get_backend( language="en-us", backend="espeak", punctuation=True, stress=True, strip=True ):
	key = f'{language}_{backend}'
	if key in cached_backends:
		return cached_backends[key]

	if backend != 'espeak':
		raise ValueError(f"unsupported phonemizer backend: {backend} (only espeak)")
	# remove-flags: no "(en)...(de)" markers around words espeak reads in another language (the dataset has none)
	phonemizer = EspeakBackend( language, preserve_punctuation=punctuation, with_stress=stress, language_switch="remove-flags")

	cached_backends[key] = phonemizer
	return phonemizer


def encode(text: str, language="auto", backend="auto", punctuation=True, stress=True, strip=True) -> list[str]:
	if language == "auto":
		language = detect_language( text )

	language = coerce_language( language )
	if language[:2] == "en":
		text = normalize_en( text )

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
	phonemes = backend.phonemize( [ text ], strip=strip )
	
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

def encode_batch(texts: list[str], language: str) -> list[str]:
	"""encode() over many texts of one language in one espeak call (the store's rephonemize)."""
	language = coerce_language( language )
	if language[:2] == "en":
		texts = [ normalize_en( t ) for t in texts ]
	elif language[:2] == "ja":
		texts = [ coerce_to_hiragana( t ) for t in texts ]
	out = _get_backend(language=language, backend="espeak").phonemize( texts, strip=True )
	if language == "cmn-latn-pinyin":
		tones = { "1": "ˇ", "2": "ˉ", "3": "ˊ", "4": "ˋ", "5": "_" }
		out = [ "".join( tones.get(c, c) for c in o ) for o in out ]
	return out

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