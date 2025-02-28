# Training Notes

Training is very dependent on:
* the quality of your dataset.
  * clean utterances and accurate transcriptions go a long way.
  * a diverse dataset in prosody and speakers help a ton.
* how much data you have.
  * training from scratch requires upwards of 15K hours at minimum.
  * training new languages from the base model simply requires maybe ~2K hours each.
* the bandwidth you quantized your audio to, as this affects the how many tokens are processed per step.
* the underlying model architecture used.
  * some models behave better than others for a unified approach, others do not.

A training paradigm that works for me is:
* setting the dataloader to sort by duration, then training until coherent speech emerges, so the model can start with the bulk of learning on small, insignificant utterances, then working its way up to larger ones.
  * ~80% of the epoch from duratio ranges 1.0seconds to 0.8seconds is good enough, as most of the training from this part is just to train the model to speak at all.
* additional training using a shuffled dataloader, as the model will be fixated towards whatever duration range it was trained under.
  * the remaining bulk is to try and have the model better adhere to the prompt as well.
* additional training for sampling per speaker, to better help diversify how well it can perform for a range of speakers, rather than just speaking itself
  * I don't think this is crucial, but speaker-based sampling seems to be a placebo if anything.

Training under `float16` (+AMP) should be fairly simple, but it's practically required to use the `deepspeed` backend.
* This is because `deepspeed` will automatically wrap the optimizer to handle training under `float16`, while the `local` backend does not do this. Training will *not* converge.
* Training under `bfloat16` does not have to worry about this.

When training from scratch, maybe 30% of the time spent training is getting coherent speech, with a loose following of the prompt. The remaining bulk of the work is getting the model to closely-er resemble the input prompt.
* an accuracy of at least 50% seems to be where coherent speech emerges.
* an accuracy of at least 68% is about where it's a good enough model that adheres to the prompt, but requires quite a lot of work to get there.

As far as typical hyperparameters go:
* as I'm using a batched dataloader, I don't have a uniform size amongst the batches, but I believe my average batch size is between 96 to 128 samples per batch (24-32 samples per GPU for 4 GPUs) per step.
* the gradient accumulation factor gets adjusted where I feel is best, where I keep it to 1 (no gradient accumulation) for the first milestone of getting coherent speech, and then ramping it up to 2 then 4 as training further goes on, to try and smooth out the gradients.
  * more effective samples per update step is technically better, but getting coherent speech as fast as possible is preferable, so prioritizing many updates until then is the goal.
  * afterwards, reducing the gradient norm is the goal, increasing the amount of samples per update step.
* as I primarily use prodigyopt, I don't need to worry about the learning rate. Sure, you can lower the `d_coef` (which the trainer will adjust in lieu of the learning rate itself), but I don't feel like it effects things moreso than just adjusting the gradient accumulation factor.

With the other "hyperparameters" such as ratios for RVQ levels, tasks, etc:
* `rvq_levels_p` to `auto` is fine. The primary level is RVQ level 0, so having it majorly represented is fine.
  * it might be needed to later prefer a more balanced distribution (such as `[0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7]`) to get rid of any confidence issues in RVQ levels 1+, but I felt naively doing this harms the RVQ 0.
* `prompt_similar_p` can be pretty much whatever > `0.5`. I've stuck with either `0.75` or `0.825` to prioritize adhering closely-er to the prompt, but still have random prompts used to help the model interanlly "model" what a speaker should sound like. In theory.

The optimizer used *mostly* doesn't matter, as AdamW seems to get moving faster, while Prodigyopt keeps things stable in the long run.
* `APOLLO` needs more testing, but seemed adequate in cursory tests
* `Muon` requires much more testing, but absolutely cannot be used for predicting tokens in place (NAR demasking), and requires `cfg.model.experimental.predict_causally=True`
  * I honestly don't think it gives good enough results from curosry tests for this application

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

## Knowledge Distillation

Performing knowledge distillation from a teacher to a student is simple. All that's needed is to reference the teacher model in under `cfg.models`, and mark `teacher: True`, and the student model will automatically reference the teacher model.

Additional hyperparameters can be tuned to what you want under `cfg.hyperparameters`, but the defaults are sane:
* `teacher_alpha`: the alpha to blend between the normal logits, and the soft targets from comparing the probability distribution from the student model to the teacher model. `0.5` works fine enough.
* `teacher_temperature`: the temperature to apply to the logits for both the student and the teacher, that is then also applied to the soft targets. `1.0` seems fine.
* `teacher_loss_fn`: the type of loss function to use. `kl` will use `kl_div` on the probability distributions, while `mse_loss` will apply to the raw logits before applying softmax. Either are fine: `kl` is commonly used, while some literature swear by `mse_loss` for a trivial gain.

# `train.py`

This script handles the VALL-E specific training code.

For the most part, this handles:
* feeding the model a batch from the dataloader
* performing evaluation / validation when requested
* unloading the `emb.qnt` model when its not needed anymore

For single GPUs, simply running `python3 -m vall_e.train --yaml="./training/config.yaml`.

For multiple GPUs, or exotic distributed training:
* with `deepspeed` backends, simply running `deepspeed --module vall_e.train --yaml="./training/config.yaml"` should handle the gory details.
* with `local` backends, simply run `torchrun --nnodes=1 --nproc-per-node={NUMOFGPUS} -m vall_e.train --yaml="./training/config.yaml"`

You can enter `save` to save the state at any time, or `quit` to save and quit training.

The `lr` command will also let you adjust the learning rate on the fly. For example: `lr 1.0e-3` will set the learning rate to `0.001`.

Some additional flags can be passed as well:
* `--eval`: only run the evaluation / validation pass, then exit afterwards.
* `--eval-random-text-prompts`: use random text prompts for the evaluation pass, rather than the provided text prompts in the dataset.