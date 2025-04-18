# Training Notes

Training is (obviously) *very* dependent on:
* the size of your dataset
  * for getting a model to speak, prioritizing lots of samples over lots of speakers is preferred
  * for getting the model to adhere to the prompt, prioritizing lots of speakers over samples is preferred
  * smaller utterances are better to train the model faster in aggregate
    * but longer utterances are still required to have the model able to output longer utterances
      * *technically* can be augmented by concatting utterances together, as long as there's some coherency between them.
* the quality of your dataset
  * an automatically transcribed / segmented is *fine* enough with some slight start/end offsets
    * if you *can* avoid relying on segmenting, go for it
  * while I feel having 100% accurate annotations goes a long way, the absolute size of a good dataset can have it robust enough from small errors
  * exotic datasets with annotated non-speech utterances (gasps, sighs, etc.) would *really* go the long way, but the current annotation stack does not account for it
  * annotating each utterance with it's top-k similar utterances through `vall_e.emb.similar` help with prompt adherence in any stage of training
* how patient you are
  * the original (`base.py`) implementation serendipitously has a cirriculum that allows for speech to realize relatively fast with EnCodec (from what I remember)
    * this is from how selecting which codebook level to train naturally "scales the loss" for higher, less important levels, and the model doesn't train each level in parallel at all
  * the new (`base_v2.py`) implementation requires lots of patience, as it seems to require 8M samples for speech to properly realize 
    * this is from how several "stabilizers" are required to train it as every sequence is inherently trained in parallel, but not the loss calculation.
* the audio codec, to an extent

Training is (not-so-obviously) not dependent on:
* the model size, to an extent
  * for the old (`base.py`) implementation, further experimentation is required, but from what I remember the smaller models don't have speech emerge as fast, while the larger size models don't seem to benefit much.
  * for the new (`base_v2.py`) implementation, it seems that model size doesn't affect quality at all, at least in the primary phase of getting it to speak.
    * the "training progression" (how the loss/accuracy/grad norm curves look) are about the exact same between the "normal" (1024 dim, 12 layers, 12 heads) size and the "half" (512 dim, 12 layers, 8 heads) size, and presumably for the "double" size (1538 dim, 24 layers, 24 heads).
      * the "half" size might actually be too small for it to have enough "capacity" to attend to all the speakers I'm training against.
      * per E2/F5's paper, a size of 1024 dim, 4x FFN, 16 heads, 24 layers might be preferable?
    * it *probably* is necessary to have a larger model to better adhere to the reference clip, but experimentation is not at the point yet to verify this.
* the audio codec, to an extent
  * for the old (`base.py`) implementation, EnCodec only seems to work well for it, as DAC might requires some hacks or patience for the higher codebook levels to train, while `nvidia/audio-codec-44khz` requires an exotic training cirriculum, assumedly.
  * for the new (`base_v2.py`), given how EnCodec and `nvidia/audio-codec-44khz` both seem to behave the same, I assume this implementation is almost agnostic to any codec (as long as RVQ/FSQ-ness is signaled proper).
  * each codec will have different cirriculum requirements and the ease for coherent speech to emerge from each levels will vary

A training paradigm that seems to work for me is to:
* set the dataloader to sort by duration, and get one to two epochs in.
  * this phase of training focuses on getting speech to emerge in the first place
  * this lets the model train on lots of smaller utterances, faster, before scaling it up to longer utterances
* when the output is adequate enough, switch to setting the dataloader to shuffle batches of durations instead
  * this phase of training focuses targeting the model's prompt adherence capabilities
  * this also benefits from the model training on a variety of durations to avoid it overfitting for the last duration set trained against
* optionally, you can sample based on speaker instead to balance out the speakers trained against, but this isn't all that necessary

Training under `float16` (+AMP) should be fairly simple, but it's practically required to use the `deepspeed` backend.
* This is because `deepspeed` will automatically wrap the optimizer to handle training under `float16` and does some extra magic for stability. The `local` backend does do loss scaling, but not the extra steps.
* Training under `bfloat16` does not have to worry about this, but I feel `bfloat16` training sessions don't have a specific training trait that `float16` does have, personally.

Previously, I had a good eye on when speech emerges from the loss and accuracy with the old (`base.py`) implementation, but the new (`base_v2.py`) implementation doesn't have that quality. Set the evaluation/validation frequency at a decent enough rate to not interrupt training so often, but not seldom enough that it's useless, and keep an ear on the samples. If it sounds one specific set of the audio is wrong, while other portions of the speech sounds fine, you might need to adjust which levels gets selected for training and/or the per-level loss scaling factors.

As far as typical hyperparameters go:
* the very initial phase of training targets:
  * 64 samples per batch (4 GPUs * 8 samples per batch)
  * an AdamW optimizer set to an `1.0e-4` learning rate
  * warm up scheduler for 2000 steps, then hold to the peak learning rate
  * these settings can definitely be tweaked, but this focuses on prioritizing amount of weight updates over raw throughput and "accurately modeling the probability density"
* later phases of training can switch to:
  * 128 samples per batch (4 GPUS * 16 samples per batch)
  * `prodigyopt` optimizer with the default `lr`/`d_coef` of 1.0
  * these settings focuses on stabilizing gradient flow, or some such.
* during the "target prompt adherence" phase, either a larger batch size or a `>1` gradient accumulation factor can be considered to further stabilize gradient flow.
* I feel given the nature of the model, there isn't much gain with minmaxing hyperparameter settings, as training progresses similarly regardless
* the pseudo-hyperparameter that matters more is the audio codec level cirriculum, where:
  * RVQ based codecs like EnCodec prioritizes the highest level the most, but as training progresses, the lower levels can be trained against more and more
  * for FSQ based codecs, shuffling between targetting the midrange and then the lows/highs helps
  * in the future, a (semi)-automatic cirriculum selector should be added to help with training, but the right cirriculum needs to be explored for
  * `prompt_similar_p` definitely needs further exploration, as it would govern model behavior, such as:
    * low values would guide the model to reconstruct a generalized representation of the speaker from the given prompt
    * high values would better align the reconstructed representation of a speaker to the prompt, but that representation won't generalize to other outputs
    * in other words, as a zero-shot model, low values would have it generalize better but weaken prompt adherence with having any clip be used, while high values would benefit from prompt adherence, but requires the user to have a better representation of what they want provided through the reference clip

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