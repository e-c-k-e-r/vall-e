# `webui.py`

A Gradio-based web UI is accessible by running `python3 -m vall_e.webui`. You can, optionally, pass:

* `--yaml=./path/to/your/config.yaml`: will load the targeted YAML
* `--model=./path/to/your/model.sft`: will load the targeted model weights
* `--listen 0.0.0.0:7860`: will set the web UI to listen to all IPs at port 7860. Replace the IP and Port to your preference.

## Inference

Synthesizing speech is simple:

* `Input Prompt`: The guiding text prompt. Each new line will be its own generated audio to be stitched together at the end.
* `Audio Input`: The reference audio for the synthesis. Under Gradio, you can trim your clip accordingly, but leaving it as-is works fine.
  - A properly trained model can inference without a prompt to generate a random voice (without even needing to generate a random prompt itself).
* `Output`: The resultant audio.
* `Inference`: Button to start generating the audio.
* `Basic Settings`: Basic sampler settings for most uses.
* `Sampler Settings`: Advanced sampler settings that are common for most text LLMs, but needs experimentation.

All the additional knobs have a description that can be correlated to the inferencing CLI flags.

Speech-To-Text phoneme transcriptions for models that support it can be done using the `Speech-to-Text` tab.

## Dataset

This tab currently only features exploring a dataset already prepared and referenced in your `config.yaml`. You can select a registered voice, and have it randomly sample an utterance.

In the future, this *should* contain the necessary niceties to process raw audio into a dataset to train/finetune through, without needing to invoke the above commands to prepare the dataset.

## Settings

So far, this only allows you to load a different model without needing to restart. The previous model should seamlessly unload, and the new one will load in place.