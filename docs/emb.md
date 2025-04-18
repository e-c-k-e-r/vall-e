# `emb/*`

This folder contains scripts to handle the text and audio data that goes in and out of the model, as well as preparing data for the dataset.

The `emb` name is a relic of the original implementation used.

## `g2p.py`

This script handles taking text of a given language, and phonemizing into IPAs.
* This is mainly an abstraction to `phonemizer`.

Some additional languages receive preprocessing/postprocessing:
  * For Japanese, text is coerced through `pykakasi` into hiragana, then phonemized, as `phonemizer` does not like kanji.
  * For Chinese (Mandarin), the tone markers are replaced with equivalent tone markers to differentiate between being numbered.

By default:
  * `espeak` is used as the backend, but other *backends* can be passed through `encode`.
  * punctuation, stress markers, and stripping are enabled by default, but *can* be disabled.
  * language for a given text is automatically deduced with `langdetect`, if language is not provided (or `auto` is passed).

To avoid memory leaking through `phonemizer`, backends and instances are cached for further reuse.

Unfortunately, there are some quirks with this method, as contextually things like "read" and "lead" do not rhyme with "said".

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
  - models at 44KHz + 8kbps will work for lower codebook levels, but higher codebook levels will ***always*** have issues
    * this seems to be inherent to the codec itself and not the model, as separate implementations have this problem
* [`nvidia/audio-codec-44khz`](https://huggingface.co/nvidia/audio-codec-44khz): boasts even better compression and quality
  - this codec employs FSQ instead of RVQ.
    * this doesn't seem to have any problems inherent to the codec itself, but instead inherent to FSQ codecs in general

#### Descript-Audio-Codec

Descript-Audio-Codec was thoroughly tested for promising much, much cleaner output audio, as this model encodes/decodes at 44.1KHz, rather than EnCodec's 24KHz.

However, due to the nature of the codec, simply throwing it at an attention-based transformer proves to be painful, as the model *heavily* suffers from noisy output in the higher half of the codebook levels.
* the solution may be to simply encode / decode with *all* codebook levels in one pass.

Ironically, testing through erroneously encoded audio (feeding 24KHz audio without upsampling to 44.1KHz) proved to have "cleaner" but bad utterances.

#### `nvidia/audio-codec-44khz`

This novel codec promises more than DAC without the difficulty to model with it.

NVIDIA's NeMo audio codec doesn't necessarily have a concrete name, but is simply referred to as `nemo` in the code. The included code under `./emb/codecs/nemo.py` is mostly copied (with attribution) from the reference implementation with additional tweaks. In the future, it would be beneficial to decouple it from NeMo's framework and its dependencies.

However, because this codec relies on FSQ (Finite Scalar Quantization) rather than RVQ (Residual Vector Quantization), each level of the codebook governs a specific band of the mel spectrum, instead of each level for RVQ governs additive levels to the final audio. Because of this, the original approach of inferencing the strongest detail, then each level predicts the weaker, next detail, is theoretically not a good fit for FSQ-based codecs.

The current approach is to, instead, encode / decode all FSQ levels within each pass. This approach seems promising, as it does not seem to exhibit the problem `descript-audio-codec` did where higher levels fail to train sufficiently enough.

## `transcribe.py`

This script primarily handles taking raw input audio, and outputting adequate metadata containing transcriptions of said audio through `whisper`.

By default, `openai/whisper-large-v3` is used through HuggingFace's `pipeline` and everything is handled automatically. The process maintains slices `whisper` thinks its best per the segments outputted, alongside the deduced language (if not specified).

One limiting factor is that transcription transcribes into normal text, rather than the IPA phonemes the model was trained against. Some flavors *may* exist, but I have yet to test them extensively (if I did ever find one).

Refer to the `__main__`'s arguments for usage details.

## `process.py`

This script handles taking raw input audio and its transcribed metadata, and outputs encoded audio (NumPy) files containing encoded audio and associated metadata.

This process can utilize sliced segments within the transcription metadata, or use the entire file's audio instead for a given utterance.

Refer to the `__main__`'s arguments for usage details.

> [!NOTE]
> If you're using this to try and split your workload over multiple process / GPUs, it is *imperative* to make sure to keep each process within its own NUMA node by prefixing with `numactl -N0 -m0`, or you'll experience bottlenecks that make processing worse off compared to just doing it with one GPU.

## `similar.py`

This script handles taking either raw input audio, or processed encoded audio, and determines the top-K similar utterances for each sample for a given speaker (or dataset).
* For raw input audio, the MFCC (Mel-frequency cepstrum coefficients) are extracted as features from the waveform, and the cosine similarities are compared against every other utterance for a given speaker.
  * This works *fine*, as this is adequately accurate and does not require a model to already exist.
* For the encoded audio, the audio codes are passed through the model's embedding, summed to one "token", and the cosine similarities are compared to score the top-K similar speakers.
  * By default, the output response embedding is used, and each codebook level is summed together to leave one sequence.
  * In theory this should be better as the model may have its own features per codebook + level, but still requires a model to already be trained.
  * The original encoding model's embeddings can also be used, or the last hidden states passed through the model, instead, but seems overkill.

When processing a dataset, this requires already having accompanying metadata generated through `vall_e.data --action=metadata --yaml=./your/training/config.yaml`.

Be *very* careful if you opt to output unsegmented and segmented utterances, as the sliced version may end up amongst the top-K similar candidates.

Refer to the `__main__`'s arguments for usage details.