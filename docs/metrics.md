# `metrics.py`

This file provides helper functions for computing objective metrics, such as word-error rate (WER), character-error rate (CER), and speaker similarity (SIM-O).

## WER / CER

Word-error rate (WER) is simply computed by transcribing the requested input, and comparing its transcription against the target transcription.

Because of issues with normalization (and not having a robust normalization stack), both transcriptions are then phonemized, then the resultant phonemes are used for error rate calculations.



## SIM-O