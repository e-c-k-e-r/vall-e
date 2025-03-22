# Model V2 Notes

This section aims to document the `_v2` class of models. Documentation here might be all over the place from having to extract findings from four weeks worth of agonizing experiments.

Unlike the original, this implementation strives to operate on *all* codebooks at once with a full 44KHz bandwidth, rather than requiring the model to operate on one codebook level at a time at 24KHz audio.

This model *might* not scale up well, as the `nemo-smaller-44khz-llama-8` brand seems to perform at a similar quality to `nemo-larger-44khz-llama-8`. While the latter had speech emerge much quicker than the former, both seem to have a problem with consistently working on various speakers unlike the previous series of models.
* The current issue seems to be it poorly following the prompted speaker, which if I remember right, required quite a few epochs to resolve in the base `ar+nar-len-llama-8` model.

## Audio Codecs

This implementation should work for *any* codec, as it seems to "work" adequately for:
* `nvidia/audio-codec-44khz`: an FSQ codec with 86 frames per second, 8 codebooks, and 1000 codes per codebook
* EnCodec: an RVQ codec with 75 frames per second, 8 codebooks, and 1024 codes per codebook
	* additional experimentation is required to ensure there's no emergent problems, but it seems fine so far
* DAC: an RVQ codec with 87 frames per second, 9 codebooks, and 1024 codes per codebook
	* additional experimentation is required to ensure the prior codebook problem doesn't emerge here too

In theory, RVQ codecs should work better, as "importance" is consolidated in levels that can be prioritized more, rather than FSQ codecs having no inherent priority (which requires all levels to be treated importantly, or some attention mechanism to derive importance).
* The underlying model could technically derive this importance itself, as it does receive the entire signal.
* The glamor of `nvidia/audio-codec-44khz` might not be so glamorous as the codebooks might be too dense for a model to easily operate on efficiently, as well as the codec's encoder/decoder being ***slow*** on ROCm.
	* in other words, DAC might be preferable as a 44KHz medium.
	* this might simply be a problem that can be "worked out" with more training time, hopefully, just as the "low confidence of higher codebook level" problem eventually works itself out.
	* this might also simply just be tied to the model's ability to follow closely to the prompt, as it seems more training does somewhat help out, and there doesn't seem to be a specific codebook that has confidence issues on severely unseen speakers.

## `AudioEncoder` / `AudioDecoder`

Because this model operates on the full audio sequence at once, extra care is required to ensure the model accurately operates on it, rather than leave it to chance that the model will inherently encode/decode from its latent space.

The `AudioEncoder` embeds each codebook level (and injects level-position embedding information), stacks it, then passes it through an MLP ( / residual feedforward network ), then weighs each level through learned weights before summing it down to one sequence.
* I feel most of this is kind of overkill, since I believe layer 0 of the underlying model could do this better, but it might also allow better tuning of the model's "encoder" with an explicit one over an inherent one.
* Attention could also be used in place of the learned weights, as different speakers *will* have different priorities in the audio spectrum, but I imagine this might end up as a learned feature that emerges within the attention heads of the underlying model itself.

The `AudioDecoder` projects the last hidden state through another feed-forward network (non-residual, with its own pre-layer norm). The decoder can be configured to either share the head for all levels, or dedicate a head for each level.
* I feel non-shared heads might also be overkill, but allows for the decoder to better-er extract the dedicated codebook level from the last hidden state.
* It might not even be necessary to use an MLP, as the model was quick to fix itself after deleting-then-shrinking the feed-forward expansion factor to try and squeeze out throughput.
	* because of this ablation, it's *probably* okay to just do per-codebook layer norm + an output head, but that experimentation is for another day.

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

Unlike the previous implementation, duration prediction is trained in parallel with the base `tts` task, where the output feature is always at the separator after the input prompt. This moves away from the kludge of treating the duration as an extra "language" task with a vocab size of `11`, and decoded autoregressively, while allowing some wiggle room in the duration as it's no longer sampled using logits.

### Pure AR

Unlike the previous implementation, this model can also operate entirely autoregressively as a causal transformer, where each step samples *all* codebooks at one code-frame.

More experimentation is needed for this modality, but seeing as the pure NAR approach works, I imagine a model can either be trained purely-autoregressively, or mixed (such as with the confusingly named `ar+nar-len`) model.

However, this modality was not trained for either models, as there seems to be some weird quirk when inferencing that's caught under CUDA, but not ROCm. This doesn't seem to "go away" with more training, unfortunately.

## Training Regimen

The `nemo-smaller-44khz-llama-8` model is a 512-dim, 12 layered, 8 headed attention-based transformer with rotary position embedding. Training was performed on four V100s with AMP+`float16` with a batch size of 8 samples per GPU, and an AdamW optimizer with adequate parameters (`1.0e-4` learning rate,  betas of `[0.8, 0.95]`, weight_decay of `0.01`, linear warmup to 5K steps before holding) for 400K steps before introducing training for duration prediction in parallel. The dataloader sorts the dataset by duration, starting from 2 seconds and ending with 8 seconds-ed utterances. Training consists of computing the loss for each codebook level non-parallely (where a level is randomly assigned to a sample per a "normal" distribution) with each loss being weighed "normal"ly, for 70% of the epoch when speech starts to emerge. Then, the model was trained to compute the loss paralelly (where all levels have the loss computed) without weighing the loss per-level. Audio quality was lacking for most speakers, as the model failed to handle all codebook levels adequately. Additional training slowly helps, but by-the-numbers metrics don't show much improvement.
* this model also had some training on my 7900XTX rig under `bfloat16`, with similar hyperparameters (a batch size of 32 for one GPU, rather than 8 samples * 4 GPUs ), as it ironically is at parity when utilizing `flash_(sdpa)` attention.

The `nemo-larger-44khz-llama-8` model is similar to its immediate predecessor, with 1024-dim, 24 layers, and 16 heads. Training is similar where the only difference is with a learning rate of `3.0e-4`.  Speech emerged quicker than its predecessor at `?`% of the epoch, but quality remains about the same.

Training of both models experienced degredation in quality periodically, where the loss will rise, spike, then climb back down. It's reasonable to assume this came from duration sorting being the cause, as the model might somehow "overfit" based on duration, as this problem disappeared when re-initializing the dataloader to instead batch samples by durations, then shuffle the batches. However, training throughput significantly dropped for the larger model.

The differences between the two models suggests there is no outright immediate benefits from scaling up as it "costs" more to train the larger model. Benefitis may be discovered through manual evaluation, which kind of predicates on the duration predictor (which wasn't added until much later into training out of neglect).

Both flavors were trained on the previously used dataset, but English only (as I did not want to risk throwing in multiple languages during the initial training session, and my patience was dwindling during the audio processing phase).

Additional tasks beyond text-to-speech (`tts`) were not trained for either models, as they're very low priority, and the implementation might have had logic to train for it gutted.

### Experimental Settings

Currently, both models are trained using these experimental flags:
```
unified_position_ids: False # per-segment position IDs

rvq_levels_p: "equal" # distribution of codebook levels to target training for
audio_level_loss_factors: "normal" # distribution of loss weights per codebook (should be "equal" when speech is confident enough)

masking_train_p: 1.0 # pure AR
masking_ratio: 0.8 # fixed mask ratio proves to be better
ignore_inputs_for_loss: True # False is not implemented 
use_segmented_attention_mask: True # restricts each section within its own section + prior section (in other words, does not let the text/prom see further into the future outside of its segment)
use_streamlined_calc_loss: True # False has no effect now

noncausal_masks: True # creates non-causal masks
resp_parallel_training: True # trains all codebook levels in parallel
len_parallel_training: True # trains length duration alongside normal training 

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
* `use_sliding_attention_mask`: this applies a sliding attention mask within each segment of the input (for example, slide within the text, slide within the prom, slide within the resp), as something said in the beginning of the utterance shouldn't affect what's aid at the end
	* however, it seems this is a detriment to the model, I imagine because the model could rely on how something sounds earlier on, even if there shouldn't be a direct causal relationship
	* this could be something that might need to be trained from the very beginning rather than early on, but training existing models does not seem to fare well
		* `nemo-smaller-llama-8` seemed to have degraded far more than `nemo-larger-llama-8` did. I suppose the head count / size might matter.

## Benefits and Caveats

To be evaluated, as additional training time is required, despite progression seemingly plateu-ing.

At a glance, compared to the prior model setup, this implementation allows for the model to better represent speech as it's able to see the entire signal and account for it in its latent space, rather than only specific levels of it.

Additionally, this implementation paves the way a ton of neat features, such as:
* live playback through autoregressive inferencing, as all codebooks are predicted for each step
	* could also be "mocked" by doing NAR-len demasking in chunks
* inherent audio upscaling, as the model is trained on a 44KHz codec
* some other features I can't recall

However, I'm not sure if the additional complexity justifies it.
* the current hurdle is that speaker similarity is ***dismal***
