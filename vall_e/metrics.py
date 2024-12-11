# handles objective metric calculations, such as WER and SIM-O

#from .emb.transcribe import transcribe
from .emb.similar import speaker_similarity_embedding
from .emb.transcribe import transcribe
from .emb.g2p import detect_language
from .data import normalize_text

import torch.nn.functional as F

from pathlib import Path
from torcheval.metrics.functional import word_error_rate

def wer( audio, reference, language="auto", **transcription_kwargs ):
	if language == "auto":
		language = detect_language( reference )

	transcription = transcribe( audio, language=language, align=False, **transcription_kwargs )["text"]

	# reference audio needs transcribing too
	if isinstance( reference, Path ):
		reference = transcribe( reference, language=language, align=False, **transcription_kwargs )["text"]

	transcription = normalize_text( transcription )
	reference = normalize_text( reference )

	return word_error_rate([transcription], [reference]).item()

def sim_o( audio, reference, **kwargs ):
	audio_emb = speaker_similarity_embedding( audio, **kwargs )
	reference_emb = speaker_similarity_embedding( reference, **kwargs )

	return F.cosine_similarity( audio_emb, reference_emb ).item()