# Text normalisation and the scoring units for WER / CER (whisper output against reference transcripts).
import re
import unicodedata

CER_LANGUAGES = ("zh", "ja", "ko")  # scored per character: no (reliable) word boundaries

# normalization code borrowed from TorToiSe TTS
# (it's not perfect but it works)


# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
	('mrs', 'misess'),
	('mr', 'mister'),
	('dr', 'doctor'),
	('st', 'saint'),
	('co', 'company'),
	('jr', 'junior'),
	('maj', 'major'),
	('gen', 'general'),
	('drs', 'doctors'),
	('rev', 'reverend'),
	('lt', 'lieutenant'),
	('hon', 'honorable'),
	('sgt', 'sergeant'),
	('capt', 'captain'),
	('esq', 'esquire'),
	('ltd', 'limited'),
	('col', 'colonel'),
	('ft', 'fort'),
]]
def normalize_abbreviations(text):
	for regex, replacement in _abbreviations:
		text = re.sub(regex, replacement, text)
	return text

def _remove_commas(m):
	return m.group(1).replace(',', '')

def _expand_decimal_point(m):
	return m.group(1).replace('.', ' point ')

def _expand_dollars(m):
	match = m.group(1)
	parts = match.split('.')
	if len(parts) > 2:
		return match + ' dollars' # Unexpected format
	dollars = int(parts[0]) if parts[0] else 0
	cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
	if dollars and cents:
		dollar_unit = 'dollar' if dollars == 1 else 'dollars'
		cent_unit = 'cent' if cents == 1 else 'cents'
		return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
	elif dollars:
		dollar_unit = 'dollar' if dollars == 1 else 'dollars'
		return '%s %s' % (dollars, dollar_unit)
	elif cents:
		cent_unit = 'cent' if cents == 1 else 'cents'
		return '%s %s' % (cents, cent_unit)
	else:
		return 'zero dollars'

# in case the current env does not have it installed, so I don't need it as a hard dependency
try:
	import inflect

	_inflect = inflect.engine()

	def _expand_ordinal(m):
		return _inflect.number_to_words(m.group(0))

	def _expand_number(m):
		num = int(m.group(0))
		if num > 1000 and num < 3000:
			if num == 2000:
				return 'two thousand'
			elif num > 2000 and num < 2010:
				return 'two thousand ' + _inflect.number_to_words(num % 100)
			elif num % 100 == 0:
				return _inflect.number_to_words(num // 100) + ' hundred'
			else:
				return _inflect.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
		else:
			return _inflect.number_to_words(num, andword='')
except Exception as e:
	_inflect = None

_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')
_whitespace_re = re.compile(r'\s+')
_end_punct_re = re.compile(r'[\.\?\!]$')
_aux_punct_re = re.compile(r'[,;:\?\.\!-]')

def normalize_numbers(text):
	text = re.sub(_comma_number_re, _remove_commas, text)
	text = re.sub(_pounds_re, r'\1 pounds', text)
	text = re.sub(_dollars_re, _expand_dollars, text)
	text = re.sub(_decimal_number_re, _expand_decimal_point, text)
	if _inflect is not None:
		text = re.sub(_ordinal_re, _expand_ordinal, text)
		text = re.sub(_number_re, _expand_number, text)
	return text

# full will do aggressive normalization, perfect for WER/CER
# not full will do basic cleaning
def normalize_text(text, language="auto", full=True):
	if full:
		text = text.lower()
		text = normalize_numbers(text) # expand numbers
		text = normalize_abbreviations(text) # expand abbreviations
		#text = re.sub(_end_punct_re, '', text) # collapse whitespace
		text = re.sub(_aux_punct_re, '', text) # collapse whitespace
		text = text.replace('"', '') # remove quotation marks
	else:
		text = normalize_numbers(text) # expand numbers
		text = normalize_abbreviations(text) # expand abbreviations
		text = re.sub(_whitespace_re, ' ', text) # collapse whitespace

	# to-do: other languages
	return text


def units(text, lang):
	"""Scoring units: words for English (the old normalisation: numbers and abbreviations expanded) and other spaced
	languages (lowercased, punctuation dropped), characters for zh / ja / ko."""
	if lang == "en":
		return normalize_text(text, language="en").split()
	text = "".join(" " if unicodedata.category(c)[0] in "PSZ" else c for c in unicodedata.normalize("NFKC", text).lower())
	if lang in CER_LANGUAGES:
		return list(re.sub(r"\s+", "", text))
	return text.split()


def edits(a, b):
	"""Levenshtein distance between two unit sequences."""
	prev = list(range(len(b) + 1))
	for i, x in enumerate(a, 1):
		cur = [i]
		for j, y in enumerate(b, 1):
			cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
		prev = cur
	return prev[-1]
