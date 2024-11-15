# Training Notes

Training is very dependent on:
* the quality of your dataset.
  * clean utterances and accurate transcriptions go a long way.
  * a diverse dataset in prosidy and speakers help a ton.
* how much data you have.
  * training from scratch requires upwards of 15K hours.
  * training new languages from the base model simply requires maybe ~2K hours each.
* the bandwidth you quantized your audio to, as this affects the how many tokens are processed per step.
* the underlying model architecture used.
  * some models behave better than others for a unified approach, others do not.

For single GPUs, simply running `python3 -m vall_e.train --yaml="./training/config.yaml`.

For multiple GPUs, or exotic distributed training:
* with `deepspeed` backends, simply running `deepspeed --module vall_e.train --yaml="./training/config.yaml"` should handle the gory details.
* with `local` backends, simply run `torchrun --nnodes=1 --nproc-per-node={NUMOFGPUS} -m vall_e.train --yaml="./training/config.yaml"`

You can enter `save` to save the state at any time, or `quit` to save and quit training.

The `lr` command will also let you adjust the learning rate on the fly. For example: `lr 1.0e-3` will set the learning rate to `0.001`.

Some additional flags can be passed as well:
* `--eval`: only run the evaluation / validation pass, then exit afterwards.
* `--eval-random-text-prompts`: use random text prompts for the evaluation pass, rather than the provided text prompts in the dataset.

A training paradigm that works for me is:
* setting the dataloader to sort by duration, then training one epoch, so the model starts with small utterances then trains to larger ones.
  * the daring can wait until coherent speech emerges, then move to the next step
* some additional training using a shuffled dataloader, as the model will be fixated towards whatever duration range it was trained under.
* additional training for sampling per speaker, to better help diversify how well it can perform for a range of speakers, rather than just speaking itself
  * I don't think this is crucial, but speaker-based sampling seems to be a huge placebo if anything.

I don't remember the exact numbers off the top of my head, but a good loss/accuracy/gradient norm to look out for when coherent speech emergies are:
* loss <3.0
* acc >0.7
* grad_norm <0.2

Training under `float16` should be fairly simple, but care is required to keep the loss scaling factor above 8K, and probably even 16K.
* At the very least for pre-trained models, low enough loss scales will irreparably fry the model, and no amount of training afterwards seems to "fix" it.
* The current DeepSpeed configuration should keep the loss scale capped to 32K, but this so far is only validated for pre-trained models.
* Training under `bfloat16` does not have to worry about this as there's no need for loss scaling, but I feel the model performs better when trained under `float16`+AMP rather than `bfloat16` (with or without AMP).

## Try Me

To quickly test if a configuration works, you can run `python -m vall_e.models.ar_nar --yaml="./data/config.yaml"`; a small trainer will overfit a provided utterance.

## Finetuning

Finetuning can be done by training the full model, or using a LoRA.

Finetuning the full model is done the same way as training a model, but be sure to have the weights in the correct spot, as if you're loading them for inferencing.

For training a LoRA, add the following block to your `config.yaml`:

```
loras:
- name : "arbitrary name" # whatever you want
  rank: 128 # dimensionality of the LoRA
  alpha: 128 # scaling factor of the LoRA
  training: True
```

And that's it. Training of the LoRA is done with the same command. Depending on the rank and alpha specified, the loss may be higher than it should, as the LoRA weights are initialized to appropriately random values. I found `rank` and `alpha` of 128 works fine.

To export your LoRA weights, run `python3 -m vall_e.export --lora --yaml="./training/config.yaml"`. You *should* be able to have the LoRA weights loaded from a training checkpoint automagically for inferencing, but export them just to be safe.

## Training Under Windows

As training under `deepspeed` and Windows is not (easily) supported, under your `config.yaml`, simply change `trainer.backend` to `local` to use the local training backend.

Creature comforts like `float16`, `amp`, and multi-GPU training *should* work under the `local` backend, but extensive testing still needs to be done to ensure it all functions.

# `train.py`

This script handles the VALL-E specific training code.

For the most part, this handles:
* feeding the model a batch from the dataloader
* performing evaluation / validation when requested
* unloading the `emb.qnt` model when its not needed anymore