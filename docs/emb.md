# `emb/*`

This folder contains scripts to handle the text and audio data that goes in and out of the model, as well as preparing data for the dataset.

The `emb` name is a relic of the original implementation used.

## `g2p.py`

This script handles taking text of a given language, and phonemizing into IPAs.
* This is mainly an abstraction to `phonemizer`.

For Japanese, text is coerced through `pykakasi` into kana, then phonemized, as `phonemizer` does not like kanji.

By default, `espeak` is used as the backend, but other *backends* can be passed through `encode`.

By default, punctuation, stress markers, and stripping are enabled by default, but *can* be disabled.

To avoid memory leaking through `phonemizer`, backends and instances are cached for further reuse.

## `qnt.py`

This script handles taking audio waveforms and encoding it as code tokens to run through the model, and code tokens outputted from the model and decoding it into raw waveforms.
* This is mainly an abstraction to the underlying quantized audio models.

Additionally, audio manipulation helper functions like `trim` and `concat` are available.

The audio backend is dependent on the model used, but by default `encodec` is the default backend with a sample rate of `24khz`.
* if requested, `vocos` is used as the decoding model, but EnCodec is still used to encode audio.

Audio does *not* need to be resampled and downmixed, as it should already be handled when being fed to the `encode` functions.

### Audio Backends

For audio backends:

* [`encodec`](https://github.com/facebookresearch/encodec): a tried-and-tested EnCodec to encode/decode audio.
* [`vocos`](https://huggingface.co/charactr/vocos-encodec-24khz): a higher quality EnCodec decoder.
  - encoding audio will use the `encodec` backend automagically, as there's no EnCodec encoder under `vocos`
* [`descript-audio-codec`](https://github.com/descriptinc/descript-audio-codec): boasts better compression and quality, but has issues with model convergence.
  - models at 24KHz + 8kbps will NOT converge in any manner.
  - models at 44KHz + 8kbps seems harder to model its "language", and the NAR side of the model suffers greatly.

## `transcribe.py`

This script handles taking raw input audio, and outputting adequate metadata containing transcriptions of said audio through `whisperX`.

The process maintains slices `whisperX` thinks its best per the segments outputted.

Refer to the `__main__`'s arguments for usage details.

## `process.py`

This script handles taking raw input audio and its transcribed metadata, and outputs encoded audio (NumPy) files containing encoded audio and associated metadata.

This process can utilize sliced segments within the transcription metadata, or use the entire file's audio instead for a given utterance.

Refer to the `__main__`'s arguments for usage details.

## `similar.py`

This script handles taking either raw input audio, or processed encoded audio, and determines the top-K similar utterances for each sample for a given speaker (or dataset).

When processing a dataset, this requires already having accompanying metadata generated through `vall_e.data --action=metadata --yaml=./your/training/config.yaml`.

Refer to the `__main__`'s arguments for usage details.