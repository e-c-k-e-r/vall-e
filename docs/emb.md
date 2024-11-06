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

### Text Tokens

Despite being an audio LM, the model still needs some form of text as the input prompt.

While it's possible to naively use raw text, it's much more beneficial to instead opt for tokenizing IPAs instead, as they're (mostly) not tied to the language itself.

For the meantime, this project depends heavily on `phonemizer` to process normal text into IPAs

In the future, a separate model that handles converting text into phonemes is preferred, but:
* this requires an extensive vocab *per language*.
* this either requires an additional model to lug around and have trained, or repurposing the existing model to perform such task.
  + The latter option does open the way of taking normal text as inputs itself, as the model should be aware enough about mapping text to IPAs.
  + This *technically* can be done, as it just requires a separate input embedding + output head per language, but training without hindering the model would be a chore.

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

#### Descript-Audio-Codec

Descript-Audio-Codec was thoroughly tested for promising much, much cleaner output audio, as this model encodes/decodes at 44.1KHz, rather than EnCodec's 24KHz.

However, due to the nature of the codec, simply throwing it at an attention-based transformer proves to be painful, as a unified AR+NAR model *heavily* suffers from noisy output in the NAR.

Ironically, testing through mal-encoded audio (feeding 24KHz audio without upsampling to 44.1KHz) proved to have "cleaner" but bad utterances.

I'm uncertain on how to remedy this, as my options are:
* train under a RetNet, if an attention-based transformer is simply the problem
* train an AR, and train a NAR, if the codec itself is at fault
* use an SSM like Mamba, if transformers entirely cannot model the codec
* train a separate model that simply converts from EnCodec to DAC

## `transcribe.py`

This script handles taking raw input audio, and outputting adequate metadata containing transcriptions of said audio through `whisperX`.

The process maintains slices `whisperX` thinks its best per the segments outputted, alongside the deduced language (if not specified).

One limiting factor is that transcription transcribes into normal text, rather than the IPA phonemes the model was trained against. Some flavors *may* exist, but I have yet to test them extensively (if I did ever find one).

Refer to the `__main__`'s arguments for usage details.

## `process.py`

This script handles taking raw input audio and its transcribed metadata, and outputs encoded audio (NumPy) files containing encoded audio and associated metadata.

This process can utilize sliced segments within the transcription metadata, or use the entire file's audio instead for a given utterance.

Refer to the `__main__`'s arguments for usage details.

## `similar.py`

This script handles taking either raw input audio, or processed encoded audio, and determines the top-K similar utterances for each sample for a given speaker (or dataset).
* For raw input audio, the MFCC (Mel-frequency cepstrum coefficients) are extracted as features from the waveform, and the cosine similarities are compared against every other utterance for a given speaker.
  * This works *fine*, as this is adequately accurate and does not require a model to already exist.
* For the encoded audio, the audio codes are passed through the model's embedding, summed to one "token", and the cosine similarities are compared to score the top-K similar speakers.
  * By default, the output response embedding is used, and each RVQ level is summed together to leave one sequence.
  * In theory this should be better as the model may have its own features per RVQ code+level, but still requires a model to already be trained.
  * The original encoding model's embeddings can also be used, or the last hidden states passed through the model, instead, but seems overkill.

When processing a dataset, this requires already having accompanying metadata generated through `vall_e.data --action=metadata --yaml=./your/training/config.yaml`.

Be *very* careful if you opt to output unsegmented and segmented utterances, as the sliced version may end up amongst the top-K similar candidates.

Refer to the `__main__`'s arguments for usage details.