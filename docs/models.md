# Model Notes

> [!NOTE]
> A rework of this page is required, as most of this information is outdated due to slightly wrong assumptions, and confusing terminology and conflations

The underlying model is a robust transformer, where:
* inputs are passed through an embedding
* the embedded inputs are then passed through each layer of the transformer (or other model type)
* the last hidden states are then passed through the output head / classifier / projection, resulting in logit probabilities to sample from.

The beauty of a transformer, I feel, is that you can easily define any task at it, and it should follow through with it very well.

The inputs are automatically sequenced in a way that a given task requires, and the outputs are handled as per the class that extends the base model.

While the original paper called for a separate AR model and a NAR model, by treating the AR and the NAR as unique tasks, you can actually train a unified model (`AR+NAR`) for effectively free, as the internal states of the two should overlap quite a lot.
* Additionally, you can even train a `NAR-len` model on top of an existing model.

## The AR (Autoregressive) Model

The AR is responsible for generating the first codebook level of the audio codes for a given output. References to "outputs from the AR" refers to this level, as it contibutes to the final waveform the most.
* Some models may refer to this level as the "coarse" level.
* The benefit of autoregressively decoding for this code is that it offers better output while also "encoding" the duration within the sequence itself, as the stop token will depend on the length of the sequence.
* The downside is that it does take most of the compute time to iterate through the sequence one step at a time.

Autoregressive training is performed by having each token predict the next token in the sequence. This is done by appending a special stop token to the input targets, then shifting the output logits over one compared to the input targets (this shift can be more than one to decode more than one token).

One way to work around the time cost is to instead decode more than one token at a time.
* In theory, for a unified AR+NAR model, this *should* be an easy task, as the model can already decode tokens in parallel.
* In reality, this isn't the case. Specifying a `cfg.model.experimental.causal_size > 1` with adequate training will have the output sound *fine* every Nth timestep and every other timestep not so fine, as the following tokens aren't predictable enough.
  + *However*, this may simply be a sampling problem, as this experiment was done with outdated ideas on how to sample the AR, and should be worth revisiting.
* VALL-E 2's paper proposes merging code sequences together into one embedded token for a speedup.

Sampling the AR does not necessarily require a specific sampling temperature, as:
* lower temperatures follow the prompt better, at the cost of variety in the outputs, and the need to either use classifier-free guidance or repetition penalty to wrangle the output.
* higher temperatures are possible, but are more prone to not adhere to the prompt.

Traditional samplers for text-gen models can apply to the AR (especially rep/len pen), but more exotic samplers (mirostat, DRY, etc.) don't seem to offer much besides serving as bandaid solutions for a lacking AR.

Compared to non-autoregressive decoding, I personally feel that autoregressive encoding offers a specific-yet-hard-to-quantify expressive quality that the NAR (and pure NAR solutions) does not offer.

## The NAR (Non-autoregressive) Model

The NAR is responsible for generating the remaining codebook levels of the audio codes for a given output. References to the "outputs from the NAR" refers to the underlying "levels" for a given waveform, as each further levels contributes to the final waveform less significantly than the previous.
* Some models may refer to this level as the "fine" level.

As decoding is done non-autoregressively, the model can process tokens "in place" and have them attended to one another in the past and future, thus speeding up output and allowing for "more accurate" outputs.

Non-autoregressive training is performed by having the input tokens from the previous codebook level predict the next level's token in place. The output logits are in the same position, and do not require further modifications as required for the AR.

One problem exhibited from a NAR is producing arfifacts ("crust") in the final waveform. I believe this is a confidence problem where the wrong token is inferred.
* Unfortunately, one solution is to simply train a separate NAR, as this should help bolster the model's NAR capabilities without the AR influencing things, as I imagine being able to both causally and parallel-ly decode tokens harms things.
  * This is backed by the used `cfg.model.experimental.rvq_levels_p` distribution affecting the model's AR capabilities, as increasing the NAR's share in training causes the AR to perform *less*.
    * However, this may be simply wrong, but checkpoints that used such distributions felt lobotomized.
* Another solution that may help is to provide two token dropout methods:
  * `token_dropout_error`: This will randomly nudge a small percentage of tokens from the prior codebook level to simulate wrong tokens being predicted.
  * `token_dropout_rate`: This will randomly mask off tokens from the prior codebook level with a mask token, to try and have the model not-strongly-rely on the given input.

Sampling from the NAR absolutely necessitates a low temperature or to be greedily sampled, as higher temperatures lead to the aforementioned artifacts in the final waveform.
* This is mostly mitigated with a proper non-causal mask, but crust still emerges at higher temperatures.

Traditional samplers do not seem to offer much effect in the output, as it seems the output from the NAR are decent enough.

### Pure NAR

The pure NAR (`NAR-len`) model is a modality that inferences audio tokens purely non-autoregressively.

However, having a pure NAR is challenging, as you need to both explicitly provide the duration and provide a "good enough" starting sequence of tokens for the initial sequence.
* The former problem is easily "solved" by training a `len` classification task.
* The latter however proves to be challenging, as generating tokens from nothing in one step is not possible (but can be done in multiple steps).
  * diffusion solves this, but requires a different underliny model architecture
  * masking to emulate diffusion noising is best working solution, but has a lot of training challenges.
    * existing solutions like Muse (text to image) and MaskGCT (text to speech) do this

The NAR-len model keeps things simple by:
* training with a fixed masking ratio (80% of the tokens are masked and trained to predict the remaining tokens)
  * [this paper](https://arxiv.org/abs/2406.05478v1) mentions a fixed ratio during training yields better results than randomly picking a masking ratio.
  * randomly picking a duration ~~is actually very ungood and harms the model during training~~ ~~actually doesn't matter much~~ matters enough to warrant sticking with a fixed rate.
    * theoretically, it should help later stages in demasking to better rely on the non-masked tokens, but who knows.
    * in reality, it seems to harm the model being able to produce decent results in fewer steps.
* not including any specific timestep embedding information
  * some solutions add in the (sinusoidal position'd) timestep embedding, either on top of the input embeddings, or as some normalization weight around the attention head (before and after).
  * it does not seem to be necessary what-so-ever to require this, especially training under a fixed masking ratio.
    * in theory, attention *could* deduce this from the amount of masked tokens vs unmasked tokens in the sequence. 
    * in reality, the model shouldn't really need to reference this anyways, as there's no reason for the model to make use of this information when it's trying to predict what *all* masked tokens should be.
* predicting the "duration" (the output audio token window) is kept within the model itself, by autoregressievly inferencing the duration for a given input prompt (text + audio).
  * the model can already "know" the duration for a given prompt already from an AR codebook level 0, by predicting when to output the stop token, so it makes sense to re-use the model for this.
  * the output length is a simple tokenized sequence where each token is a base-10 digit.
    * it could be in any base, but it's simple to just treat each token ID as a digit, then cast the string to an int.
    * this could literally also not be relying on an AR sequence to predict.
  * some checkpoints of the model seems to adhere well to outputting silence at the end if the requested duration exceeds the actual duration.
    * this seems to only happen for models that erroneously causally attend to tokens in the `NAR-len`.
* inferencing is a simple loop that simply takes the best masked-off k tokens per step, and remasks the remaining.

Because the model already leverages the magic of attention to derive phoneme-alignment, such annotations are still not required (but they probably help with a naive sampler).

In theory, demasking for the NAR's codebook level 0 can also be applied to the remaining codebook levels to further improve the output from the remaining levels.
* this isn't necessary as the model already has a strong enough relationship between the prompt, the prior levels, and the targeted level.
* this is technically already offered with `cfg.model.experimental.token_dropout_rate` which mirrors masking, but experimentation has not been done to a large degree.
* there is a bit of a problem with properly implementing this, as the tokens aren't predicting themselves.
  * it may be a simple thing to implement anyways.

It is ***crucial*** to:
* avoid re-masking tokens that are already "good" enough (this can easily be done by "banning" them in the scoring process)
  * without this, you ***will*** get stuttering and unaligned utterances. I do not know why this is such a big problem but I imagine this "interleaves" many different sequences between each step.
  * (although token remasking shows that this isn't a strict requirement)
* use unfiltered/unprocessed logit scores:
  * not that crucial, but helps stability, by using which part of the sequence was "confident enough" to keep.
* use a CFG strength of at least 2 (or a prefix)
  * the output falls apart completely without this.

It is not required to train a model from scratch to use this modality, as training from existing weights works just as well, if not better (as it can piggyback off the original model).
* additional training is still required to help confidence issues and to condition the model to not fall apart for longer durations.

## Embeddings (and Classifiers)

The "magic" of subjugating a transformer for audio use lies within the ensemble of the embeddings. This is necessary as each piece of a sequence is fundamentally different, but a HF-compatible model can get away with treating each sequence as separate ranges within a total token sequence.

With attention-based transformers, most embeddings can serve as a token itself and have the attention mechanism attend to it.

Other solutions such as TorToiSe makes use of additional embeddings/classifiers for each portion of the sequence as well.

### Classifiers

Classifiers are the final output head / projection layer that processes the last hidden states of a model into a probability distribution for each token. 

Out of paranoia, each head is split for each macro-task (codebook level, `stt`, and `len`), even though the core half of the model's training was with a single output head.
* It also helps with not needing to do some tricks by setting unwanted tokens to `-inf`.

### Text Embeddings

The input text phonemes (or output for STT) are passed through an embedding head (`text`), similar to how a normal text LLM would. Nothing fancy is required, as it's very straightforward.

Technically, due to how the audio embeddings are implemented, it's possible to offer "language specific" text embeddings, rather than one unified IPA-based embedding + a language embedding (`lang`).
* Such an implementation can instead inference from normal text rather than IPA phonemes, as language-specific pure text embeddings can be trained.
  * This is because some arbitrary first `n` layers of the model *might* instead handle encoding the input prompt embeddings. It's easy to take an existing model and train it on raw text tokens alongside the IPA phonemes as an input.

These embeddings *could* instead be added on top of the input prompt embedding instead of serving as additional tasks (similar to injecting position embeddings), but additional experimentation is required to see if the model both can work under this and/or benefits from this.

These embeddings can also be substituted out for a "text semantic" embedding, rather than tokenized phonemes, as the text conditioning input.
* Additionally, methods like [BLT](https://github.com/facebookresearch/blt) can replace this instead, as patching the audio portion wouldn't gain much benefit due to it already being quantized audio.

#### Language Embedding

This embedding provides the requested language for the model to be aware of.

This *mostly* isn't necessary, but VALL-E X's paper mentions needing a token for the language itself, and other solutions like XTTS2 provides a language token as well.

In reality, this seems to help govern the accent / general mannerisms associated with that language.
* For examples:
  * prompting French or German with the output language set to `en` will give typical foreigner speech of trying to speak a language they don't know.
  * prompting a Japanese speaker with the output language set to `ko` or `zh` will offer little changes to the spoken language (at least no nuance I can hear as an EOP).
* Consequently, since this does tie to accents more, ***extreme*** attention is to be paid to the dialects being trained against, instead of naively grouping, say, all of Spanish to one language code.
  * unfortunately, this does mean that audio annotated as English is dialect/accent-agnostic, per the dataset.

Some checkpoints of the model needs this for cross-lingual output, but the current checkpoints of the model doesn't seem to do this due to the attention heads deriving the language/accent from the phoneme sequences themselves rather than the language token due to a careless oversight.

#### Tone Embedding

This embedding *should* provide information on the tone for the model to output the utterance in.

Should, since I do not actually make use of this anywhere, and the model is not trained against any tones. I would need to annotate my dataset based on tones *and* pick which tones I do want.

This should most definitely help the model identify tone strongly even without needing to annotate for it, but it does an adequate job already with maintaining tone from a given input prompt.

I imagine, like language/accent, this gets derived from the phoneme sequence itself rather than a guidance token.

### Audio Embeddings

However, due to the nature of the encoded audio, embedding the audio tokens requires the dark arts, as we use audio both as an input prompt (`prom`) for guidance, and as an output response (`resp`).

As EnCodec encodes audio across eight codebooks (and DAC's 44Khz audio under nine codebooks), our audio is encoded under a 2D space, rather than a simple 1D space like text does. Because of this, we require embeddings for *every* codebook level, effectively giving eight embedding heads for audio.
* Technically, this can be stored within a unified embedding head, but each layer is offset by 1024 (the number of tokens).

For the `prom` embedding, we can simply use each embedding for each layer. Each embedding level maps to its respective codebook level.

However, the `resp` requires some extra care, as the model needs to both causally (AR) and parallel-ly (NAR) decode tokens.
* The first embedding level pertains to codebook level 0 for the AR (`AR:0:0`) or NAR (`NAR:0:0`).
  * This embedding predicts tokens within its own embedding.
* The remaining embedding levels maps to codebook level 0 + n for the NAR (`NAR:L-1:L`).
  * In other words, embedding level 1 => codebook level 0, embedding level 2 => codebook level 1, etc...
* I believe this is required because the model encodes which task to perform (rather than the attention heads), and which tokens to predict (rather than the classifiers)
  * In other words, each embedding needs to be separated based on what tokens they do predict.

The `prom` and `resp` are split since, in theory, it helps the model know better what audio to source from, and what audio is part of the output sequence. In theory.
* The `text` embedding's robustness not only for reuse between each codebook level, but for the `stt` task as well is a mystery.

Finally, the model *may* then sum each embedding level back down to one sequence, as defined under `cfg.model.experimental.audio_embedding_sums`.
* The resulant sum is not normalized by the length.
* It's not a requirement, as the model can still function only "seeing" the required codebook level.
* However, it *may* help to have the model being able to "see" prior levels, as one codebook level might depend on the prior level.
  * This is mostly dependent on the underlying audio model being used, which would depend on how each residual is defined.
* A model not trained with summing embeddings can enable it without much impact, but a model trained on summing embeddings cannot go in the other way without further training.
  * It *could* be beneficial to train a model under mixed modes, but requires experimentation.
  * The reference model was trained originally without summing, then trained with summing.

Additionally, it's *technically* possible to instead use the embeddings from the model used to encode the audio (for example, EnCodec's embeddings), but in theory this may exclude specific features the model has encoded within the embeddings.

Either embeddings can be used to compute utterance similarity scores, as per `vall_e.emb.similarity` for utterance similarities.
* I need to compare if this can be used as well for speaker similarities.
* The current implementation makes use of the `resp` embeddings for this, but the `proms` might be used instead (experimentation is needed for this).

#### Codebook Level Embedding

This embedding hints what the target codebook level of the audio codes is being targetted. This embedding is not required, but seems some architectures (Mamba) requires this.

This *may* replace needing separate embeddings for each codebook level, but experimentation is required to test this claim.

### Tasks

The base model handles processing inputs into token sequences, per the requested task assigned to each input in a batch.

Most sequences follow a `<text><codebook level><language><prompt><output>` sequence, but some tasks will receive the prompt as a list of tensors, instead.

The nitty gritty of how each task is implemented is documented under [./docs/data.md](/docs/data.md).

#### Text-to-Speech

The primary zero-shot text-to-speech synthesis `tts` task takes in a requested text transcript, a piece of reference audio, and then outputs the response audio of the utterance saying the prompted transcript.

The model primarily functions in a zero-shot setting, where it does not need a guiding prefix, but few-shotting is possible through manual intervention.
* I believe the original VALL-E paper refers to this more as `VALL-E Continuous`, while some other TTS solutions follow this method by transcribing the input audio prompt as well.
* Guidiance prefixing is offered in the implementation, but right now is only exposed under "rolling context/prefix" through the web UI / CLI (where the previous segment is used as the prefix for the next).

Additional tasks are implemented in this project, but ***are yet to be trained for*** in the reference model (as some tasks require additional compute-cost).

##### Noise Suppression

This task `ns` aims to suppress or remove noise from the input audio.

In practice, this task is already implemented by providing the input audio to denoise, and having the input transcription be the transcription of the input audio. The output isn't 1:1 exact in terms of prosody and delivery, but it's close.

I imagine training for this task will better help the model understand what is noise and what isn't, and can better strongly-er map utterances from the input audio prompt to use in the output, delivering better prompt adherance.
* This also might help serve in helping the model identify effects applied to an utterance, and being able to maintain it in normal `tts` tasks, such as reverb or the audio quality itself (the "acoustic environment").

This task can be briefly trained for decent results in-post.

##### Speech Removal

This task `sr` aims to remove speech from a given audio, effectively serving as the reverse of denoising.

As state above, this should help the model better identify what is noise and what isn't.

This task can be briefly trained for decent results in-post.

##### Target Speech Extraction

This task `tse` aims to "extract" an utterance from audio containing other speakers, effective diarizing an utterance.

I imagine training for this task will better help the model "target" the prompted input audio and adhere to it, but this task moreso appears to be solely for the task itself, rather than help the model itself.

##### Clean Speech Editing

This task `cse` aims to modify a portion of a given utterance, effectively editing it.

I imaginie training for this task *might* help the model better map to the input prompt utterances to the output response, but I don't expect for the effects to be strong enough; it definitely is a task that is for the task itself.

###### Noisy Speech Editing

This task `nse` is effectively the same as `cse`, but under a noisy condition.

#### Length Prediction

The length predictor `len` task is required for a pure NAR model.

This task will naively output a zero, then the length in base-10, followed by a stop token.

This works because the model can already derive the length of a sequence when autoregressively decoding through the probability of emitting a `<stop>` token.

#### Speech-to-Text

The speech-To-text `stt` task transcribes a given piece of audio, by taking an input encoded audio, and outputting the text transcription.

However, due to the model being trained on phonemes, the resultant output is the phonemes itself.

The primary benefit of this task is to provide a fast way to directly transcribe audio into the phonemes used annotate the dataset itself, but at the moment the reference model isn't accurate enough to rely on this.
* The other problem is it's very hard to validate this, as the output isn't in English, and requires processing through the model again to verify the transciption.

This task will follow a reverse sequence of `<audio><language><codebook level><output>`.

#### Phonemize / Un-Phonemize

The `phn` task phonemizes raw text and outputs the corresponding IPA phonemes.

The `un-phn` task does the opposite: it'll take IPA phonemes and outputs the text that would phonemize into it.

Currently, `phn` works *okay*, while `un-phn` does not work at all.

## Emergent Behavior

The model can be prompted in creative ways to yield some interesting behaviors:
* prompting without an input audio prompt will have the model generate a random voice ~~at the "cost" of some unintelligible utterance at the beginning of the output response (despite doing no promptless training)~~.
  * classifier-free-guidance-aware training does fix this, but this property emerges without it.
  * the AR is much better with this property, as the `NAR-len` gets crusty sometimes as it will keep demasking on crust.
* prompting with an input text prompt being the transcription of the input audio prompt will have the response follow very closely to the input prompt  (despite not doing input=output training).
  * this should allow for easy transcription editing without much fuss.
  * the `NAR-len` greatly exhibits this property, although it sometimes does keep any noise in the background.
  * extra care is required when doing this, as some checkpoints of the model will degrade completely the moment the prompt can't be directly referenced.
* training without a language token will have the model derive the target language/accent from the phoneme sequence itself (it is a language model after all)
* voice conversion is *possible* through demasking with the source prompt as the mask, but the current inferencing mechanism yields crust at the end of the output

# `models/*`

This folder contains scripts relating to models and code for VALL-E use, from the wrapping model to the underlying arch.

## `models/lora.py`

This script implements Low-Ranking Adapters, to allow for cheaper and easier finetuning of existing modules.

At the moment, two approaches are offered, through replacing `nn.Linear` outright, or parameterizing a `nn.Linear`. The latter is used by default(?).

## `models/base.py`

This script implements the core underlying model for VALL-E. This handle:
* storing its settings and features, and initializing the right modules
* processing inputs into a proper input string
* orchestrates running text and audio through the respective embeddings
* generating the right padding, masking, and position IDs to feed the underlying arch (if requested)
* removes padding from the logits
* handles performing loss calculation, both as a whole or in individual pieces, both autoregressively and non-autoregressively
* handles sampling through the logits through samplers provided through `./vall_e/samplers.py`, both autoregressively and non-autoregressively.

This script aims to implement everything as required per VALL-E agnostically, to allow for different implementations to contain little extra code.

A very naive implementation of using the model can be found under the `__main__` invocation.

### `models/base_v2.py`

This script implements a newer model aimed to sample *all* codebooks for a given step.

Due to major enough differences, this code is segregated from the original `models/base.py` to not break things further.

## `models/ar_nar.py`

This script implements VALL-E as a unified autoregressive and non-autoregressive model, where codebook level 0 is inferenced autoregressively, the remaining levels are infereneced non-autoregressively, if requested.
* Since one model can be trained AR-ly and NAR-ly, codebook level 0 can also be trained non-autoregressively with diffusion-like masking.

For training, this model handles preparing the batch provided through the dataloader according to a randomly sampled targetted codebook level.

For inferencing, this will dynamically inference depending on the arguments provided.

## `models/arch/*`

This folder contains scripts, I've either written myself or properly attributed to, that provide or modify existing modules of a given model.

As the core of VALL-E makes use of a language model, various LLM architectures can be supported and slotted in. Currently supported LLM architectures:

* `llama`: using HF transformer's LLaMa implementation for its attention-based transformer, boasting RoPE and other improvements.
  + I aim to utilize this for the foundational model, as I get to leverage a bunch of things tailored for LLaMA (and converting to them is rather easy).
* `mixtral`: using HF transformer's Mixtral implementation for its attention-based transformer, also utilizing its MoE implementation.
* `bitnet`: using [this](https://github.com/kyegomez/BitNet/) implementation of BitNet's transformer.
  - Setting `cfg.optimizers.bitnet=True` will make use of BitNet's linear implementation.
* `transformer`: a basic attention-based transformer implementation, with attention heads + feed forwards.
* `retnet`: using [TorchScale's RetNet](https://github.com/microsoft/torchscale/blob/main/torchscale/architecture/retnet.py) implementation, a retention-based approach can be used instead.
  - Its implementation for MoE can also be utilized.
* `retnet-hf`: using [syncdoth/RetNet](https://github.com/syncdoth/RetNet) with a HuggingFace-compatible RetNet model
  - has an inference penality, and MoE is not implemented.
* `mamba`: using [state-spaces/mamba](https://github.com/state-spaces/mamba) (needs to mature)
  - ***really hard*** to have a unified AR and NAR model
  - inference penalty makes it a really hard sell, despite the loss already being a low 3 after a short amount of samples processed

The wide support for various backends is solely while I try and figure out which is the "best" for a core foundation model.

### `models/arch/llama.py`

This script contains its own copy of the LLaMA provided through `transformers` with its own modifications and independence from any updates that may break it.

A bulk of it pertains to modifying `LlamaAttention` and detecting available attention mechanisms, allowing for using different attention mechanisms:
* `torch.nn.functional.scaled_dot_product_attention`-based attention:
  * `math`: torch's SDPA's `math` kernel
  * `mem_efficient`: torch's SDPA's memory efficient (`xformers` adjacent) kernel
  * `cudnn`: torch's SDPA's `cudnn` kernel
  * `flash_(sdpa)`: torch's SDPA's flash attention kernel
* internal implementations of external attention backends:
  * `xformers`: [facebookresearch/xformers](https://github.com/facebookresearch/xformers/)'s memory efficient attention
  * `flash_attn`: uses the available `flash_attn` package (including `flash_attn==1.0.9` through a funny wrapper)
  * `flash_attn_v100`: uses [ZRayZzz/flash-attention-v100](https://github.com/ZRayZzz/flash-attention-v100/)'s Flash Attention for Volta (but doesn't work currently)
  * `fused_attn`: uses an implementation using `triton` (tested on my 7900XTX and V100s), but seems to introduce errors when used to train after a while
  * `sageattn`: uses [SageAttention](https://github.com/thu-ml/SageAttention).
    * training under this is untested, but dropout is not applied (yet).
  * `default`: uses the naive path for the internal implementation (used for attention-debugging purposed)
* `transformers` Llama\*Attention implementations:
  * `eager`: default `LlamaAttention`
  * `sdpa`: integrated `LlamaSdpaAttention` attention model
  * `flash_attention_2`: integrated `LlamaFlashAttetion2` attention model
* `auto`: determine the best fit from the above

Modifications to `LlamaModel` is also provided to implement LayerSkip-aware training and a very naive self-speculative decoding.

#### ROCm Flash Attention

[ROCm/flash-attention](https://github.com/ROCm/flash-attention) currently does not support Navi3 cards (gfx11xx), so first-class support for Flash Attention is a bit of a mess on Navi3. Using the `howiejay/navi_support` branch can get inference support, but not training support (due to some error being thrown during the backwards pass) by:
* edit `/opt/rocm/include/hip/amd_detail/amd_hip_bf16.h`:
```
  #if defined(__HIPCC_RTC__)
  #define __HOST_DEVICE__ __device__ static
  #else
  #include <climits>
  #define __HOST_DEVICE__ __host__ __device__ static inline
  #endif
```
* install with `pip install -U git+https://github.com/ROCm/flash-attention@howiejay/navi_support --no-build-isolation`

Later versions of PyTorch with ROCm natively supports full (both inferencing and training) Flash Attention through SDPA's interface. To use it, just set the model's attention to `flash_(sdpa)`

### `models/arch/mamba.py`

This script modifies modules of Mamba, to allow it to play nicely with my existing code.

If I rememer right, it just simply provides gradient checkpointing.

### `models/arch/mixtral.py`

Like `llama.py`, this provides modifications to Mixtral through `transformers`. However, most of the niceties from `llama.py` are not available here as it's not the core backend.

Primarily, this is to address a bug with batch sizes > 1, and to use a different attention mechanism.
* to-do: this is out of date from `llama.py`'s modified attention class.

### `models/arch/retnet.py`

This provides modification to RetNet, mostly to allow for gradient checkpointing.

### `models/arch/transformer.py`

This provides the original implementation's implementation of a transformer.

### `models/arch/attention/*`

This folder contains specific attention mechanisms.

Currently, only `fused.py` is provided, which implements fused attention through Triton.

Attributions are noted at the top of the respective file(s).