# `inference.py`

This script handles everything the higher level functions of inferencing the model for various tasks for the end user.

## Synthesis

To synthesize speech: `python -m vall_e <text> <ref_path> <out_path> --yaml=<yaml_path>` (or `--model=<model_path>`)

Some additional flags you can pass are:
* `--language`: specifies the language for phonemizing the text, and helps guide inferencing when the model is trained against that language.
* `--task`: task to perform. Defaults to `tts`, but accepts `stt` for transcriptions.
* `--max-ar-steps`: maximum steps for inferencing through the AR model. Each second is 75 steps.
* `--device`: device to use (default: `cuda`, examples: `cuda:0`, `cuda:1`, `cpu`)
* `--ar-temp`: sampling temperature to use for the AR pass. During experimentation, `0.95` provides the most consistent output, but values close to it works fine.
* `--nar-temp`: sampling temperature to use for the NAR pass. During experimentation, the lower value, the better. Set to `0` to enable greedy sampling.
* `--input-prompt-length`: the maximum duration the input prompt can be (~6 seconds is fine, longer durations lead to slower generations for "better" accuracy, as long as the model was trained against such input prompt durations)

And some experimental sampling flags you can use too (your mileage will ***definitely*** vary, but most of these are bandaids for a bad AR):
* `--input-prompt-prefix`: (AR only) treats the input prompt as the initial response prefix, but...
  * the transcription of the prompt needs to be in the input text prompt.
  * doesn't perform all that well (I belive the model needs to be trained a bit on this, as `tts-c`).
* `--min-ar-temp`: triggers the dynamic temperature pathway, adjusting the temperature based on the confidence of the best token. Acceptable values are between `[0.0, (n)ar-temp)`.
  + This simply uplifts the [original implementation](https://github.com/kalomaze/koboldcpp/blob/dynamic-temp/llama.cpp#L5132) to perform it.
  + **!**NOTE**!**: This does not seem to resolve any issues with setting too high/low of a temperature. The right values are yet to be found.
* `--top-p`: limits the sampling pool to top sum of values that equal `P`% probability in the probability distribution.
* `--top-k`: limits the sampling pool to the top `K` values in the probability distribution.
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
* `--layer-skip` enables early-exit layer skipping if the model is confident enough (for compatible models)
* `--layer-skip-exit-layer`: maximum layer to use
* `--layer-skip-entropy-threshold`: the maximum the logits' entropy (confidence) needs to be before exiting early
* `--layer-skip-varentropy-threshold`: the maximum the logits' varentropy (confidence spread) needs to be before exiting early
* `--refine-on-stop`: (AR only) uses the last steps' logits for the entire final output sequence, rather than the step-by-step iterative sequence.
  + This needs experimenting with to see if there's any downside.
  + to-do: compare the probability scores with the original output sequence, and pick the best one.

### Speech-to-Text

The `ar+nar-tts+stt-llama-8` model has received additional training for a speech-to-text task against EnCodec-encoded audio.

Currently, the model only transcribes back into the IPA phonemes it was trained against, as an additional model or external program is required to translate the IPA phonemes back into text.
* this does make a model that can phonemize text, and unphonemize text, more desirable in the future to replace espeak (having an additional task to handle this requires additional embeddings, output heads, and possible harm to the model as actual text is not a modality the model is trained on).