# Model V2 Notes

This section aims to document the `_v2` class of models. Documentation here might be all over the place from having to extract findings from several weeks worth of agonizing experiments and quirks.

Unlike the original, this implementation strives to operate on *all* codebooks at once with a full 44KHz bandwidth, rather than requiring the model to operate on one codebook level at a time at 24KHz audio.

Sample weights can be found [here](https://huggingface.co/ecker/vall-e/) under the `nemo-` prefix.

## Audio Codecs

This implementation should work for *any* codec, as it seems to "work" adequately for:
* `nvidia/audio-codec-44khz`: an FSQ codec with 86 frames per second, 8 codebooks, and 1000 codes per codebook
* EnCodec: an RVQ codec with 75 frames per second, 8 codebooks, and 1024 codes per codebook
	* additional experimentation is required to ensure there's no emergent problems, but it seems fine so far
* DAC: an RVQ codec with 87 frames per second, 9 codebooks, and 1024 codes per codebook
	* additional experimentation is required to ensure the prior codebook problem doesn't emerge here too

In theory, RVQ codecs should work better, as "importance" is consolidated in levels that can be prioritized more, rather than FSQ codecs having no inherent priority (which requires all levels to be treated importantly, or some attention mechanism to derive importance).
* The underlying model could technically derive this importance itself, as it does receive the entire signal.

## `AudioEncoder` / `AudioDecoder`

Because this model operates on the full audio sequence at once, extra care is required to ensure the model accurately operates on it, rather than leave it to chance that the model will inherently encode/decode from its latent space.

The `AudioEncoder` embeds each codebook level (and injects level-position embedding information), stacks it, then passes it through an MLP ( / residual feedforward network ), then weighs each level through learned weights before summing it down to one sequence.
* I feel most of this is kind of overkill, since I believe layer 0 of the underlying model could do this better, but it might also allow better tuning of the model's "encoder" with an explicit one over an inherent one.
* Attention could also be used in place of the learned weights, as different speakers *will* have different priorities in the audio spectrum, but I imagine this might end up as a learned feature that emerges within the attention heads of the underlying model itself.
* As an MLP is employed, the initial embedding dimension can be entirely decoupled from the underlying model's width. This allows for easy drop-in from the embeddings of the audio codec utilized.

The `AudioDecoder` projects the last hidden state through another feed-forward network (non-residual, with its own pre-layer norm). The decoder can be configured to either share the head for all levels, or dedicate a head for each level.
* I feel non-shared heads might also be overkill, but allows for the decoder to better-er extract the dedicated codebook level from the last hidden state.
* It might not even be necessary to use an MLP, as the model was quick to fix itself after deleting-then-shrinking the feed-forward expansion factor to try and squeeze out throughput.
	* because of this ablation, it's *probably* okay to just do per-codebook layer norm + an output head, but that experimentation is for another day.

### Ablations

For RVQ codecs, such as EnCodec and DAC, the `AudioEncoder.level_weights` can be ignored entirely without any problem.
* FSQ codecs might require this, as one test suggests removing this caused a nasty problem.

For any codec, the `AudioEncoder.norm` can be omitted, as it doesn't make much sense to perform layer normalization pre-FFN when the input is just the embedding + codebook positioning embeddings.
* it *might* instead work when applying it to the input into the FFN rather than the input entirely, or applying it post-FFN on the residual connection.

Both `AudioEncoder.ffn` and `AudioDecoder.ffn` can have its expansion size adjusted and re-trained without much of an impact (for example, to downsize).

### `ResidualAudioEncoder/Decoder`

The implementation also includes an encoder/decoder targeted for residual codecs, but real-world testing shows that it does not perform anywhere near as well as the FSQ-targeted encoder/decoder setup.

This might be simply from it relying on cross-attention to deduce codebook level importance, rather than using an bone-standard feed-forward network with learned weighting of the codebooks (since the codebooks should always have a fixed relationship).

## Modalities

The same core modalities are supported when inferencing all codebooks in parallel.

While the model *can* still be trained as a hybrid AR/NAR, the reference `nemo-*-llama-8` family of models are trained purely as a masked NAR transformer, due to:
* the previous implementation nicely handled storing embeddings of different modalities, while this implementation does not have an inherent mechanism to do so without relying on some form of additional weights somewhere
	* this is *probably* because the embeddings signal to the modal whether to predict tokens in place or predict the next token in the sequence
	* *technically* the NAR sequences can be trained to predict the next token instead, but that's completely untested and may cause problems
* even training with a causal (triangle) attention mask lobotomizes the model severely
	* I don't think it's worth the compute at the moment to brute force through it
	* the original implementation of NAR-demasking showed the model was too flawed to be used when naively using a causal (triangle) mask, so I would not want to tempt fate to ruin the model
* an autoregressive decoder *could* be emulated with decoding in chunks
	* in addition to some additional lines of code to add in (which would probably just re-use the "rolling context" feature), an attention mask similar to sliding attention is required I imagine

### Pure NAR

Like the previous implementation, this model can operate entirely non-autoregressively (and with non-causal attention) as a masked transformer. The demasking inference loop is the same as the previous implementation, where each demasking step can mask off an entire timestep on the sum of the logit scores, or independently (where each level has its own mask).

Quasi-similarly to the previous implementation, duration prediction is trained through an explicit task, but unlike the previous implementation, this does not need to be autoregressively inferenced. By making use of my findings with a classifier-based OCR model, duration prediction can be done with one "token" and reshaping it into several digits for the final logits.
* Instead of a discrete, logit based output, it's possible to instead output a raw float to correspond to the seconds and train using `mse_loss` (or maybe `kl_div`), but experimentation shows that it's quite a pickle to train, even with weighing its loss down considerably.
* This *could* be trained in parallel using clever tricks with the attention mask, but a regression in the model/code suggests it's not worth wrangling for this feature.

#### Attention

As suggested from the prior implementation, attention needs to be paid to the attention mask used, as it's quite easy to have the model degrade from a silent problem.

A naive, fully non-causal attention just works, and while it seems a little unintuitive to have the input prompt attend to the output, I imagine the model does some bookkeeping in the input portion of the prompt. By constraining the input to not attend to the output, the model also grows constrained in its performance.

Prior (failed) experimentation with exotic masks include:
* a segmented mask, where each segment can only attend to itself and the prior segment.
* sliding attention, where each segment can only attend to its own window instead of its entire segment.

### Pure AR

Unlike the previous implementation, this model can also operate entirely autoregressively as a causal transformer, where each step samples *all* codebooks at one code-frame.

More experimentation is needed for this modality, but seeing as the pure NAR approach works, I imagine a model can either be trained purely-autoregressively, or mixed (such as with the confusingly named `ar+nar-len`) model.

However, this modality was not trained for either models, as there seems to be some weird quirk when inferencing that's caught under CUDA, but not ROCm. This doesn't seem to "go away" with more training, unfortunately.
* Additionally, I'm under the impression that separate `resps_embs` are required for causal/non-causal sequences, as the previous implementation inherently has this split.

However however, this modality doesn't seem to be necessary as the benefits of a pure NAR model outweighs the benefits of a pure AR one.

## Training Regimen

The `nemo-smaller-44khz-llama-8` model is a 512-dim, 12 layered, 8 headed attention-based transformer with rotary position embedding. Training was performed on four V100s with AMP+`float16` with a batch size of 8 samples per GPU, and an AdamW optimizer with adequate parameters (`1.0e-4` learning rate,  betas of `[0.8, 0.95]`, weight_decay of `0.01`, linear warmup to 5K steps before holding) for 400K steps before introducing training for duration prediction in parallel. The dataloader sorts the dataset by duration, starting from 2 seconds and ending with 8 seconds-ed utterances. Training consists of computing the loss for each codebook level non-parallely (where a level is randomly assigned to a sample per a "normal" distribution) with each loss being weighed "normal"ly, for 70% of the epoch when speech starts to emerge. Then, the model was trained to compute the loss paralelly (where all levels have the loss computed) without weighing the loss per-level. Audio quality was lacking for most speakers, as the model failed to handle all codebook levels adequately. Additional training slowly helps, but by-the-numbers metrics don't show much improvement.
* this model also had ~~some~~ plenty of training on my 7900XTX rig under `bfloat16`, with similar hyperparameters (a batch size of 32 for one GPU, rather than 8 samples * 4 GPUs ), as it ironically is at parity for throughput when utilizing `flash_(sdpa)` attention.
* it's reasonable to assume that a lot of the nitty gritty like LR warmup and slowly introducing features are entirely unnecessary
* the model *may* benefit from setting the dataloader to a speaker-based one, so it can balance different speakers.
* due to some form of regression, training under bfloat16 (even with AMP) will cause the gradient norm to slowly grow along with the loss.

The `nemo-larger-44khz-llama-8` model is similar to its immediate predecessor, with 1024-dim, 24 layers, and 16 heads. Training is similar where the only difference is with a learning rate of `3.0e-4`.  Speech emerged quicker than its predecessor at `?`% of the epoch, but quality remains about the same.
* increasing the de-facto batch size and lowering the learning rate seems to be necessary to edge out improvements in speaker similarity.
* Ironically, I imagine that training under bfloat16 also causes stability problems.

Training of both models experienced degradation in quality periodically, where the loss will rise, spike, then climb back down. It's reasonable to assume this came from duration sorting being the cause, as the model might somehow "overfit" based on duration, as this problem disappeared when re-initializing the dataloader to instead batch samples by durations, then shuffle the batches. However, training throughput significantly dropped for the larger model.
* Training should *probably* only have the dataloader duration-ordered until speech does emerge, then train an epoch with shuffled durations.
* Speaker similarity does improve when balancing for it without risking overfitment when it doesn't cull any speakers, but there's still inherent problems that suggest a capacity limit being reached.

The differences between the two models start to emerge on how the model can generalize. The smaller model seems to have trouble handling a variety of speakers and no inherent way of inferencing duration, while the larger model reaches its "capacity" much much later in training.

Both flavors were trained on the previously used dataset, but English-only utterances until speech was quasi-consistent.
* Additional languages and the remaining 8 seconds to 12 seconds were re-introduced into the dataset. Non-English language performance needs to be evaluated, but it seems *fine*.

Additional tasks beyond text-to-speech (such as `ns`, `sr`, `stt`) were not trained for either models, as they're very low priority, and the implementation might have had logic to train for it gutted.
* `ns` and `sr` are being experimented with, but training is a ***huge*** pain as CPU-inferencing through the NAC is required per the dataloader.
	* this could be worked around by baking a noised dataset rather than doing it at runtime (because EnCodec can easily do it, but not DAC/NeMO)

### Experimental Settings

Currently, both models are trained using these experimental flags:
```
unified_position_ids: False # per-segment position IDs

rvq_levels_p: "equal" # distribution of codebook levels to target training for
audio_level_loss_factors: "normal" # distribution of loss weights per codebook (should be "equal" when speech is confident enough)

masking_train_p: 1.0 # pure AR
masking_ratio: 0.8 # fixed mask ratio proves to be better
ignore_inputs_for_loss: True # False is not implemented 
use_segmented_attention_mask: False #
use_streamlined_calc_loss: True # False has no effect now
len_loss_factor: 0.0001 # start with the default for a while to not let duration training overpower the model, then gradually increase this (but this may only be required when introducing duration training on existing weights)

noncausal_masks: True # creates non-causal masks
resp_parallel_training: True # trains all codebook levels in parallel
len_parallel_training: False # trains length duration alongside normal training 

cfg_cond_dropout_p: 0.02 # was originally 0.3, but I think it's too much after a while
cfg_prom_dropout_p: 0.01 # was originally 0.2

use_raw_text_p: 0.1 # I don't know what's a good value, and I haven't tried inferencing with raw text yet
```

These settings should be avoided:
* `predict_causally`: forces the model to always predict the next token instead of the token in place, but untested for an actual model
	* the original NAR-demasking experiment suggests this probably is fine, but I don't want to take any risks
* `logit_normalization`: *should* have some regularlization or whatever for logits, but in reality lobotomizes inferencing output.
* `parallel_attention_mask_dropout`: this governs the rate of flipping to a causal (triangle) mask for training
	* there's *some* reason to do this ablation, but it ruins the model (but the model can easily recover if erroneously trained with this)
	* the model might eventually train itself to work around this, or it might need to be aware of this from the beginning, but it's not something to toy with.
* `use_segmented_attention_mask`: training metrics suggests this is fine, but real world usage shows it's not.
* `use_sliding_attention_mask`: this applies a sliding attention mask within each segment of the input (for example, slide within the text, slide within the prom, slide within the resp), as something said in the beginning of the utterance shouldn't affect what's aid at the end
	* however, it's possible this is a detriment itself, but further experimentation is needed
* `len_parallel_training`: this uses a clever quirk with how attention works to train duration prediction alongside normal TTS tasks
	* however, it seems there's a regression that caused this to stop working consistently
	* disabling this falls back to explicitly training a `len` task (like the old implementation)

## Samplers

Sampling code is effectively the same, with the twist of instead outputting the logits for all codebooks at `dim=0`. Returned scores are before applying sampler settings, as the unfiltered scores are preferred when doing demasking.

The NAR-demasking step will account for this automatically, and has dials and knobs to adjust. Per the web UI:
* `Masked Only`: will update scores for previously masked tokens, and unmasked tokens will not get remasked.
	* This seems *fine* when backported to the old implementation, but the new one does not like doing this.
* `Flattened`: will average out the scores for all codebook levels at a given timestep, effectively masking across all codebooks for a given timestep, rather than have it independently masked
	* This doesn't seem to have any changes, however. Greedy sampling with this on and off seems to always produce the same output, but there might be a bug.
* `Remask`: will remask a random amount of previously unmasked tokens to keep them "updated" to the rest of the waveform by some ratio (that I forget but I think it's half the masking ratio)
	* This offers some differences, but further evaluation is required to see if this is more of a good than a bad.

Basic sampler modifiers like `top-k`/`top-p` need to be evaluated but are available. Exotic samplers from the prior implementation are not included as they're bandaids applied to a modality that does not benefit from it.

## Benefits and Caveats

To be evaluated thoroughly.
* The model seems pretty quick, even for the large model.
* The smaller model seems small enough for CPU-only inferencing
	* Despite its lacking all-around zero-shot performance, it could be perfectly fine for finetuning.

At a glance, compared to the prior model setup, this implementation allows for the model to better represent speech as it's able to see the entire signal and account for it in its latent space, rather than only operate on specific levels of it at a time.

Additionally, this implementation paves the way a ton of neat features, such as:
* live playback through autoregressive inferencing, as all codebooks are predicted for each step
	* could also be "mocked" by doing NAR-len demasking in chunks, although this may require sliding window attention
* inherent audio upscaling, as the model is trained on a 44KHz codec
* some other features I can't recall

However, output leaves a lot to be desired:
* despite what literature suggests, an FSQ codec is heavily favored against with the current approach
	* each codebook's importance is effectively dependent on the speaker itself, so even having priority be a "learned" parameter is tough
	* RVQ codec don't have this problem as each level will always have the same type of importance (so much so that `AudioEncoder.level_weights` can be ignored for RVQ-codec-based models)
* this architecture does not remove the inherent problem DAC-based models have, where the higher codebooks contribute too much noise
* despite most of the dataset being resampled to 44KHz before encoding, the output audio has some inherent quality benefits (over the prior implementation's EnCodec based model) when it doesn't exhibit the below problem
* both the small and the large model seemed to have hit a "capacity" limit
	* the "confidence" problem of the prior implementation seems to have emerged even for typical speakers
* some other quirks and emergent behaviors inherent to the model I'm not aware of / can't recall
	* such as the demasking sampler loop being quite particular