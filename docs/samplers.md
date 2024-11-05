# `sampler.py`

This script contains all the samplers used during inferencing.

While I do expose these samplers for end-user use, I don't like to rely on these, as exotic samplers are always bandaids to the underlying model.

Most of these sampler functions do what's written on the tin, but for clarity:

## Samplers

When sampling, the output logits are picked for sampling according to the current inference mode. For the AR, only the last token (or last `causal_size` tokens) are used for sampling, while the NAR relies on the previous RVQ level's sequence to determine how many tokens to sample in parallel.

As the model is trained more, low temperatures are preferred over high temperatures for the AR, while greedy sampling is almost always preferred for the NAR.

Greedy sampling is enabled when the sampling temperature is <= 0, where the most likely token is picked.

### Repetition Penalty

This function (`reptition_penalize`) applies a penalty to target logits to avoid repetitive output.

This is implemented by iterating through a list of past tokens, and penalizing that token's probability score by the requested amount.

An optional value can also be passed to factor in how far away that token is.

Implicitly, this is only limited to 75 tokens in the past (one second of audio under EnCodec), and will apply more than once.

For low temperatures, this is almost necessary, as no rep-pen will have the output be garbled or a mess, and very low rep-pen will have unstable output.

### Length Penalty

This function (`length_penalize`) applies a penalty to the audio stop token (or any other specific token) based on the current length of the sequence.

This can be either a negative or a positive, to restrain or inhibit the stop token from appearing.

### Ban Tokens

This function (`ban_tokens`) bans a token from appearing.

Since this is an audio LM, there's no useful purpose for this. 

However, for some models, this is useful for banning the stop token used for the AR, when sampling output from the NAR, if the classifier / LM head / output projections are shared between the two.

### Top-K / Top-P

This function (`top_k_top_p_filtering`) filters the logits to only allow the top-K probability of tokens to be sampled, and/or the top-P probable tokens to be sampled.

This may be helpful with higher temperatured sampling to offer some variety, but not allow outputs to be *too* chaotic, in theory.

### Min-P

This function (`min_p_filtering`) filters out tokens that are under the min-P% probability.

### Dynamic Temperature

This function (`dynamic_temperature`) implements an early version of dynamic temperature per [this external PR](https://github.com/LostRuins/koboldcpp/pull/464).

To reiterate, this is an early implementation, as I recall it changing after I have already implemented this.

In theory, this allows the model to sample under higher temperatures when able, but I still need to test this the more the model receives training.

### Mirostat

This function (`mirostat_sample`) implements mirostat sampling. From what I understand, this modifies the logits based on "surprise" factor.

This may be an early implementation, as this was implemented a while back.

This *sometimes* helps the output a lot for some states of the model, but I don't try to rely on this too much.

### DRY Sampling

This function (`dry_sampling`) implements DRY sampling, a replacement to naive repetition penalizing.

I'm still not too sure what's so good about it, since it just seems like rep-pen with a different coat of paint, and for audio it doesn't seem to be too helpful?

### Entropix

This function (`sample_entropix`) implements entropix sampling, a sampler that aids in Chain-of-Thought for text LLMs by adjusting sampling parameters according to the logits and attentions' entropy and varentropy.

The huge caveat is that this requires tuning the parameters and thresholds per model, and in testing it doesn't seem like the metrics are consistent enough to rely on this. Acquiring the right attention scores is pretty much a dark art in its own right, as it does not map perfectly to naive attention, much less any other attention mechanism, under `transformers`' LLaMA.

Additionally, one state requires injecting a CoT token, which doesn't have an analog in the audio domain. 

However, this does seem to serve as a good basis to expand upon this and sample according to the entropy/varentropy of the model's current state.