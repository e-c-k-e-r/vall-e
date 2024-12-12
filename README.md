<p align="center">
<img src="./vall-e.png" width="500px"></img>
</p>

# VALL'E

An unofficial PyTorch implementation of [VALL-E](https://vall-e-demo.ecker.tech/) (last updated: `2024.12.11`), utilizing the [EnCodec](https://github.com/facebookresearch/encodec) encoder/decoder.

A demo is available on HuggingFace [here](https://huggingface.co/spaces/ecker/vall-e).

## Requirements

Besides a working PyTorch environment, the only hard requirement is [`espeak-ng`](https://github.com/espeak-ng/espeak-ng/) for phonemizing text:
- Linux users can consult their package managers on installing `espeak`/`espeak-ng`.
- Windows users are required to install [`espeak-ng`](https://github.com/espeak-ng/espeak-ng/releases/tag/1.51#Assets).
  + additionally, you may be required to set the `PHONEMIZER_ESPEAK_LIBRARY` environment variable to specify the path to `libespeak-ng.dll`.
- In the future, an internal homebrew to replace this would be fantastic.

## Install

Simply run `pip install git+https://git.ecker.tech/mrq/vall-e` or `pip install git+https://github.com/e-c-k-e-r/vall-e`.

This repo is tested under Python versions `3.10.9`, `3.11.3`, and `3.12.3`.

## Pre-Trained Model

Pre-trained weights can be acquired from
* [here](https://huggingface.co/ecker/vall-e) or automatically when either inferencing or running the web UI.
* `./scripts/setup.sh`, a script to setup a proper environment and download the weights. This will also automatically create a `venv`.
* when inferencing, either through the web UI or CLI, if no model is passed, the default model will download automatically instead, and should automatically update.

## Documentation

The provided documentation under [./docs/](./docs/) should provide thorough coverage over most, if not all, of this project.

Markdown files should correspond directly to their respective file or folder under `./vall_e/`.