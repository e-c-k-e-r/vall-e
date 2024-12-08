# `inference.py`

This script handles everything the higher level functions of inferencing the model for various tasks for the end user.

For invoking this model in another Python package, refer to `webui.py` and `demo.py` on how to use this outside of this scope.

`__main__.py` invokes this according to the below arguments.

## Synthesis

To synthesize speech: `python -m vall_e <text> <ref_path> <out_path> --yaml=<yaml_path>` (or `--model=<model_path>`)

Some additional flags you can pass are:
* `--language`: specifies the language for guiding guide inferencing when the model is trained against that language. Use `auto` to automatically deduce this.
* `--text-language`: the language to phonemize the input text under. Leave blank to tie it to the above value.
* `--task`: task to perform. Defaults to `tts`, but accepts `stt` for transcriptions.
* `--max-duration`: maximum token-duration for inferencing through the AR aspect of the model. Every second corresponds to 75 steps.
* `--max-steps`: maximum steps for inferencing through the NAR-len aspect of the model.
* `--device`: device to use (default: `cuda`, examples: `cuda:0`, `cuda:1`, `cpu`)
* `--ar-temperature`: sampling temperature to use for the AR/NAR pass. 0 enables greedy sampling.
  * For the AR, ~1.0 is *fine*, but lowering the temperature adheres better to the prosody of the input prompt.
  * For the AR, low temperatures require a repetition penalty to prevent outputs from degenerating.
  * For the NAR, greedy sampling is best, but can be raised to 0.2.
* `--input-prompt-length`: the duration of the input prompt (~6 seconds is fine, longer durations lead to slower generations for "better" accuracy). 0 does not repeat/trim.
  * If a prompt is shorter than the given duration, it's repeated to the duration size.

And some experimental sampling flags you can use too (your mileage will ***definitely*** vary, but most of these are bandaids for a bad AR):
* `--input-prompt-prefix`: (AR only) treats the input prompt as the initial response prefix, but...
  * the transcription of the prompt needs to be in the input text prompt.
  * doesn't perform all that well (I belive the model needs to be trained a bit on this, as `tts-c`).
* `--min-temperature`: triggers the dynamic temperature pathway, adjusting the temperature based on the confidence of the best token. Acceptable values are between `[0.0, (n)ar-temperature)`.
  + This simply uplifts the [original implementation](https://github.com/kalomaze/koboldcpp/blob/dynamic-temp/llama.cpp#L5132) to perform it.
* `--top-p`: limits the sampling pool to top sum of values that equal `P`% probability in the probability distribution.
* `--top-k`: limits the sampling pool to the top `K` values in the probability distribution.
* `--min-p`: only logits above `P`% probability are considered for sampling (or something, I'm still unsure how this differs from top-p).
* `--repetition-penalty`: modifies the probability of tokens if they have appeared before. In the context of audio generation, this is a very iffy parameter to use.
  * `--repetition-penalty-decay`: modifies the above factor applied to scale based on how far away it is in the past sequence.
* `--length-penalty`: (AR only) modifies the probability of the stop token based on the current sequence length. This is ***very*** finnicky due to the AR already being well correlated with the length.
* `--beam-width`: (AR only) specifies the number of branches to search through for beam sampling.
  + This is a very naive implementation that's effectively just greedy sampling across `B` spaces.
* `--mirostat-tau`: (AR only) the "surprise value" when performing mirostat sampling.
  + This simply uplifts the [original implementation](https://github.com/basusourya/mirostat/blob/master/mirostat.py) to perform it.
  + **!**NOTE**!**: This is incompatible with beam search sampling (for the meantime at least).
  * `--mirostat-eta`: (AR only) the "learning rate" during mirostat sampling applied to the maximum surprise.
* `--dry-multiplier`: (AR only) performs DRY sampling, the scalar factor.
  * `--dry-base`: (AR only) for DRY sampling, the base of the exponent factor.
  * `--dry-allowed-length`: (AR only) for DRY sampling, the window to perform DRY sampling within.

Some arguments are able to be prefixed with `ar-` and `nar-` to only use that setting for its respective pass. At the moment through the CLI, this includes:
* `temperature`

### Speech-to-Text

The `ar+nar-tts+stt-llama-8` (now the reference model) model has received additional training for a speech-to-text task against EnCodec-encoded audio.

Currently, the model only transcribes back into the IPA phonemes it was trained against, as an additional model or external program is required to translate the IPA phonemes back into text.
* this does make a model that can phonemize text, and unphonemize text, more desirable in the future to replace espeak (having an additional task to handle this requires additional embeddings, output heads, and possible harm to the model as actual text is not a modality the model is trained on).
* it seems to really want to only transcribe the first sentence for a given utterance. I imagine this is simply a problem with how it was trained.