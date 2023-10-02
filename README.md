<p align="center">
<img src="./vall-e.png" width="500px"></img>
</p>

# VALL'E

An unofficial PyTorch implementation of [VALL-E](https://valle-demo.github.io/), utilizing the [EnCodec](https://github.com/facebookresearch/encodec) encoder/decoder.

[Main Repo](https://git.ecker.tech/mrq/vall-e) | [GitHub Mirror](https://github.com/e-c-k-e-r/vall-e/) | [HuggingFace Space](https://huggingface.co/spaces/ecker/vall-e)

> **Note** This README is still quite a disorganized mess.

## Requirements

* [`DeepSpeed`](https://github.com/microsoft/DeepSpeed#requirements):
  - DeepSpeed training is Linux only. Installation under Windows should ignore trying to install DeepSpeed.
  - If your config YAML has the training backend set to `deepspeed`, you will need to have a GPU that DeepSpeed has developed and tested against, as well as a CUDA or ROCm compiler pre-installed to install this package.

* [`espeak-ng`](https://github.com/espeak-ng/espeak-ng/):
  - For phonemizing text, this repo requires `espeak`/`espeak-ng` installed.
  - Linux users can consult their package managers on installing `espeak`/`espeak-ng`.
  - Windows users are required to install [`espeak-ng`](https://github.com/espeak-ng/espeak-ng/releases/tag/1.51#Assets).
    + additionally, you may be require dto set the `PHONEMIZER_ESPEAK_LIBRARY` environment variable to specify the path to `libespeak-ng.dll`.

## Install

Simply run `pip install git+https://git.ecker.tech/mrq/vall-e`.

I've tested this repo under Python versions `3.10.9` and `3.11.3`.

## Try Me

### Online

A HuggingFace space hosting the code and models can be found [here](https://huggingface.co/spaces/ecker/vall-e).

### Local

To quickly try it out, you can run `python -m vall_e.models.ar_nar yaml="./data/config.yaml"`

Each model file has a barebones trainer and inference routine.

## Pre-Trained Model

My pre-trained weights can be acquired from [here](https://huggingface.co/ecker/vall-e).

For example:

```
git lfs clone --exclude "*.h5" https://huggingface.co/ecker/vall-e ./data/ # remove the '--exclude "*.h5"' if you wish to also download the libre dataset.
python -m vall_e "The birch canoe slid on the smooth planks." "./path/to/an/utterance.wav" --out-path="./output.wav" yaml="./data/config.yaml"
```

## Train

Training is very dependent on:
* the quality of your dataset.
* how much data you have.
* the bandwidth you quantized your audio to.

### Pre-Processed Dataset

A "libre" dataset can be found [here](https://huggingface.co/ecker/vall-e/blob/main/data.h5). Simply place it in the same folder as your `config.yaml`, and ensure its `dataset.use_hdf5` is set to `True`.

### Leverage Your Own Dataset

> **Note** It is highly recommended to utilize [mrq/ai-voice-cloning](https://git.ecker.tech/mrq/ai-voice-cloning) with `--tts-backend="vall-e"` to handle transcription and dataset preparations.

1. Put your data into a folder, e.g. `./data/custom`. Audio files should be named with the suffix `.wav` and text files with `.txt`.

2. Quantize the data: `python -m vall_e.emb.qnt ./data/custom`

3. Generate phonemes based on the text: `python -m vall_e.emb.g2p ./data/custom`

4. Customize your configuration and define the dataset by modifying `./data/config.yaml`. Refer to `./vall_e/config.py` for details. If you want to choose between different model presets, check `./vall_e/models/__init__.py`.

If you're interested in creating an HDF5 copy of your dataset, simply invoke: `python -m vall_e.data --action='hdf5' yaml='./data/config.yaml'`

5. Train the AR and NAR models using the following scripts: `python -m vall_e.train yaml=./data/config.yaml`
* If distributing your training (for example, multi-GPU), use `deepspeed --module vall_e.train yaml="./data/config.yaml"`

You may quit your training any time by just entering `quit` in your CLI. The latest checkpoint will be automatically saved.

### Dataset Formats

Two dataset formats are supported:
* the standard way:
  - data is stored under `${speaker}/${id}.phn.txt` and `${speaker}/${id}.qnt.pt`
* using an HDF5 dataset:
  - you can convert from the standard way with the following command: `python3 -m vall_e.data yaml="./path/to/your/config.yaml"`
  - this will shove everything into a single HDF5 file and store some metadata alongside (for now, the symbol map generated, and text/audio lengths)
  - be sure to also define `use_hdf5` in your config YAML.

### Plotting Metrics

Included is a helper script to parse the training metrics. Simply invoke it with, for example: `python3 -m vall_e.plot yaml="./training/valle/config.yaml"`

You can specify what X and Y labels you want to plot against by passing `--xs tokens_processed --ys loss stats.acc`

### Notices

#### Training Under Windows

As training under `deepspeed` and Windows is not supported, under your `config.yaml`, simply change `trainer.backend` to `local` to use the local training backend.

Keep in mind that creature comforts like distributed training or `float16` training cannot be verified as working at the moment.

#### Training on Low-VRAM Cards

During experimentation, I've found I can comfortably train on a 4070Ti (12GiB VRAM) with `trainer.deepspeed.compression_training` enabled with both the AR and NAR at a batch size of 16.

VRAM use is also predicated on your dataset; a mix of large and small utterances will cause VRAM usage to spike and can trigger OOM conditions during the backwards pass if you are not careful.

Additionally, under Windows, I managed to finetune the AR on my 2060 (6GiB VRAM) with a batch size of 8 (although, with the card as a secondary GPU).

## Export

To export the models, run: `python -m vall_e.export yaml=./data/config.yaml`.

This will export the latest checkpoints, for example, under `./data/ckpt/ar-retnet-2/fp32.pth` and `./data/ckpt/nar-retnet-2/fp32.pth`, to be loaded on any system with PyTorch, and will include additional metadata, such as the symmap used, and training stats.

## Synthesis

To synthesize speech, invoke either (if exported the models): `python -m vall_e <text> <ref_path> <out_path> --ar-ckpt ./models/ar.pt --nar-ckpt ./models/nar.pt` or `python -m vall_e <text> <ref_path> <out_path> yaml=<yaml_path>`

Some additional flags you can pass are:
* `--max-ar-steps`: maximum steps for inferencing through the AR model. Each second is 75 steps.
* `--device`: device to use (default: `cuda`, examples: `cuda:0`, `cuda:1`, `cpu`)
* `--ar-temp`: sampling temperature to use for the AR pass. During experimentation, `0.95` provides the most consistent output, but values close to it works file.
* `--nar-temp`: sampling temperature to use for the NAR pass. During experimentation, `0.2` provides clean output, but values upward of `0.6` seems fine too.

And some experimental sampling flags you can use too (your mileage will ***definitely*** vary):
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
* `--mirostat-eta`: (Ar only) the "learning rate" during mirostat sampling applied to the maximum surprise.

## To-Do

* train and release a ***good*** model.
* clean up the README, and document, document, document onto the wiki.
* extend to multiple languages ([VALL-E X](https://arxiv.org/abs/2303.03926)) and addditional tasks ([SpeechX](https://arxiv.org/abs/2308.06873)).
* improve throughput:
  - properly utilize RetNet's recurrent forward / chunkwise forward passes
  - utilize an approach similar to [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa/) with additional heads for decoding N+1, N+2, N+3 AR tokens
    + this requires a properly trained AR, however.
* work around issues with extending context past what's trained (despite RetNet's retention allegedly being able to defeat this):
  - "sliding" AR input, such as have the context a fixed length.
    + may require additional training to be aware of this, might not.
    + may require some phoneme/codec alignment, might not.

## Notices and Citations

Unless otherwise credited/noted, this repository is [licensed](LICENSE) under AGPLv3.

- [EnCodec](https://github.com/facebookresearch/encodec) is licensed under CC-BY-NC 4.0. If you use the code to generate audio quantization or perform decoding, it is important to adhere to the terms of their license.

- This implementation was originally based on [enhuiz/vall-e](https://github.com/enhuiz/vall-e), but has been heavily, heavily modified over time.

```bibtex
@article{wang2023neural,
  title={Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers},
  author={Wang, Chengyi and Chen, Sanyuan and Wu, Yu and Zhang, Ziqiang and Zhou, Long and Liu, Shujie and Chen, Zhuo and Liu, Yanqing and Wang, Huaming and Li, Jinyu and others},
  journal={arXiv preprint arXiv:2301.02111},
  year={2023}
}
```

```bibtex
@article{defossez2022highfi,
  title={High Fidelity Neural Audio Compression},
  author={Défossez, Alexandre and Copet, Jade and Synnaeve, Gabriel and Adi, Yossi},
  journal={arXiv preprint arXiv:2210.13438},
  year={2022}
}
```
