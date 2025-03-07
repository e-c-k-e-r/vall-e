# handles objective metric calculations, such as WER and SIM-O

#from .emb.transcribe import transcribe
from .emb.similar import speaker_similarity_embedding
from .emb.transcribe import transcribe
from .emb.g2p import detect_language, coerce_to_hiragana, encode
from .data import normalize_text

import torch.nn.functional as F

from pathlib import Path
from torcheval.metrics.functional import word_error_rate
from torchmetrics.functional.text import char_error_rate

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def wer( audio, reference, language="auto", phonemize=True, normalize=True, **transcription_kwargs ):
	if language == "auto":
		language = detect_language( reference )

	transcription = transcribe( audio, language=language, align=False, **transcription_kwargs )
	
	if language == "auto":
		language = transcription["language"]
	
	transcription = transcription["text"]

	# reference audio needs transcribing too
	if isinstance( reference, Path ):
		reference = transcribe( reference, language=language, align=False, **transcription_kwargs )["text"]

	if language == "ja":
		transcription = coerce_to_hiragana( transcription )
		reference = coerce_to_hiragana( reference )

	if phonemize:
		transcription = encode( transcription, language=language )
		reference = encode( reference, language=language )
	elif normalize:
		transcription = normalize_text( transcription, language=language )
		reference = normalize_text( reference, language=language )

	wer_score = word_error_rate([transcription], [reference]).item()
	# un-normalize
	wer_score *= len(reference.split())
	
	cer_score = char_error_rate([transcription], [reference]).item()
	# un-normalize
	cer_score *= len(reference)

	return wer_score, cer_score

def sim_o( audio, reference, **kwargs ):
	audio_emb = speaker_similarity_embedding( audio, **kwargs )
	reference_emb = speaker_similarity_embedding( reference, **kwargs )

	return F.cosine_similarity( audio_emb, reference_emb, dim=-1 ).item()