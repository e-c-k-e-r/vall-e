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
	* this might also simply just be tied to the model's ability to follow closely to the prompt, as it seems more training does somewhat help

## `AudioEncoder` / `AudioDecoder`

Because this model operates on the full audio sequence at once, extra care is required to ensure the model accurately operates on it, rather than leave it to chance that the model will inherently encode/decode from its latent space.

The `AudioEncoder` embeds each codebook level (and injects level-position embedding information), stacks it, then passes it through an MLP ( / residual feedforward network ), then weighs each level through learned weights before summing it down to one sequence.
* I feel most of this is kind of overkill, since I believe layer 0 of the underlying model could do this better, but it might also allow better tuning of the model's "encoder" with an explicit one over an inherent one.
* Attention could also be used in place of the learned weights, as some speakers *could* prioritize different codebooks levels for FSQ sequences.

The `AudioDecoder` projects the last hidden state through another feed-forward network (non-residual, with its own pre-layer norm). The decoder can be configured to either share the head for all levels, or dedicate a head for each level.
* I feel non-shared heads might also be overkill, but allows for the decoder to better-er extract the dedicated codebook level from the last hidden state.
* It might not even be necessary to use an MLP, as the model was quick to fix itself after deleting-then-shrinking the feed-forward expansion factor to try and squeeze out throughput.

### `ResidualAudioEncoder/Decoder`

The implementation also includes an encoder/decoder targeted for residual codecs, but real-world testing shows that it does not perform anywhere near as well as the FSQ-targeted encoder/decoder setup.

This might be simply from it relying on cross-attention to deduce codebook level importance, rather than using an bone-standard feed-forward network with learned weighting of the codebooks (since the codebooks should always have a fixed relationship).

## Pure NAR

Like the previous implementation, this model can operate entirely non-autoregressively (and with non-causal attention) as a masked transformer. The demasking inference loop is the same as the previous implementation, where each demasking step can mask off an entire timestep on the sum of the logit scores, or independently (where each level has its own mask).

Unlike the previous implementation, duration prediction is trained in parallel with the base `tts` task, where the output feature is always at the separator after the input prompt. This moves away from the kludge of treating the duration as an extra "language" task with a vocab size of `11`, and decoded autoregressively.

## Pure AR

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

## Benefits and Caveats

To be evaluated, as additional training time is required, despite progression seemingly plateu-ing.

At a glance, compared to the prior model setup, this implementation allows for the model to better represent speech as it's able to see the entire signal and account for it in its latent space, rather than only specific levels of it.

Additionally, this implementation paves the way a ton of neat features, such as:
* live playback through autoregressive inferencing, as all codebooks are predicted for each step
	* could also be "mocked" by doing NAR-len demasking in chunks
* inherent audio upscaling, as the model is trained on a 44KHz codec

However, I'm not sure if the additional complexity justifies it.
* the current hurdle is that speaker similarity is ***dismal***