# `metrics.py`

This file provides helper functions for computing objective metrics, such as word-error rate (WER), character-error rate (CER), phoneme-error rate (PER), and speaker similarity (SIM-O).

## WER / CER

Word-error rate (WER) is simply computed by transcribing the requested input, and comparing its transcription against the target transcription.
* The transcription is cleaned up and normalized to account for inconsistencies between transcriptions with `openai/whisper-large-v3` with the nuances of English.
* Languages without spaces between words (Chinese, Japanese) should not rely on this, and instead rely on the CER.

Character-error rate (CER) does the same thing as WER, but on a character basis rather than a word basis.

Phoneme-error rate (PER) does the same thing as CER, but on the phonemized transcription instead. As this is a speech model, this metric is more correct than the prior metrics, but this isn't a universal metric for comparison, as most models don't report this.

All rates are un-normalized because I think that's the right way to go about it? Papers aren't clear that they do this, but the error rates are even more unusually low without this.

## SIM-O

Speaker similarity (SIM-O) is computed by obtaining the embedding of each speaker (the output audio and the input prompt), and computing the cosine similarity between those two embeddings.

These embeddings are obtained through a finetune of WavLM-large geared towards speaker verification.