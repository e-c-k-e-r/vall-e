# Model Notes

The underlying model is a robust transformer, where:
* inputs are passed through an embedding
* the embedded inputs are then passed through each layer of the transformer (or other model type)
* the last hidden states are then passed through the output head / classifier / projection, resulting in logit probabilities to sample from.

The beauty of a transformer, I feel, is that you can easily define any task at it, and it should follow through with it very well.

The inputs are sequenced in a way that the given task requires automatically, and the outputs are handled as per the class that extends the base model.

While the original paper called for a separate AR model and a NAR model, and by treating the AR and the NAR as unique tasks, you can actually train a unified model for effectively free, as the internal states of the two should overlap quite a lot.

## The AR (Autoregressive) Model

The AR is responsible for generating the first RVQ level of the audio codes for a given output. References to "outputs from the AR" refers to this level, as it contibutes to the final waveform the most.
* The benefit of autoregressively decoding for this code is that it offers better output while also "encoding" the duration within the sequence itself, as the stop token will depend on the length of the sequence.
* The downside is that it does take most of the compute time to iterate through the sequence one step at a time.

Autoregressive training is performed by having each token predict the next token in the sequence. This is done by appending a special stop token to the input targets, then shifting the output logits over one compared to the input targets (this shift can be more than one to decode more than one token).

One way to work around the time cost is to instead decode more than one token at a time.
* In theory, for a unified AR+NAR model, this *should* be an easy task, as the model can already decode tokens in parallel.
* In reality, this isn't the case. Specifying a `cfg.model.experimental.causal_size > 1`  will have the output sound *fine* every Nth timestep, as the following tokens aren't predictable enough.
  + *However*, this may simply be a sampling problem, as this experiment was done with outdated ideas on how to sample the AR, and should be worth revisiting.
* VALL-E 2's paper proposes merging code sequences together into one embedded token for a speedup, but their solution seems rather complex to warrant a fundamental retrain.

I personally feel that autoregressive encoding offers a specific-yet-hard-to-quantify expressive quality that the NAR (and pure NAR solutions) does not offer, but further testing is required to substantiate the claim.

## The NAR (Non-autoregressive) Model

The NAR is responsible for generating the remaining RVQ levels of the audio codes for a given output. References to the "outputs from the NAR" refers to the underlying "levels" for a given waveform, as each further levels contributes to the final waveform less significantly than the previous.

As decoding is done non-autoregressively, the model can process tokens "in place" and have them attended to one another in the past and future, thus speeding up output and allowing for "more accurate" outputs.

Non-autoregressive trainng is performed by having the input tokens from the previous RVQ level predict the next level's token in place. The output logits are in the same position, and do not require further modifications as required for the AR.

One problem exhibited from a NAR is producing arfifacts ("crust") in the final waveform. I believe this is a confidence problem where the wrong token is inferred.
* Unfortunately, one solution is to simply train a separate NAR, as this should help bolster the model's NAR capabilities without the AR influencing things, as I imagine being able to both causally and parallel-ly decode tokens harms things.
  * This is backed by the used `cfg.model.experimental.rvq_levels_p` distribution affecting the model's AR capabilities, as increasing the NAR's share in training causes the AR to perform *less*.
    * However, this may be simply wrong, but checkpoints that used such distributions felt lobotomized.
* Another solution that may help is to provide two token dropout methods:
  * `token_dropout_error`: This will randomly nudge a small percentage of tokens from the prior RVQ level to simulate wrong tokens being predicted.
  * `token_dropout_rate`: This will randomly mask off tokens from the prior RVQ level with a mask token, to try and have the model not-strongly-rely on the given input.

### Pure NAR

The pure NAR (`nar-len`) model is a model-type that inferences audio tokens purely non-autoregressively. Despite being called a pure NAR, duration is then inferred by autoregressively decoding for its length (as the AR+NAR model shows that you can mix both types).

However, having a pure NAR is challenging, as you need to both explicitly provide the duration and provide a "good enough" starting sequence of tokens for the initial sequence.
* The former problem is easily "solved" by training a `len` inferencing task, where the given input predicts the requested duration for a given utterance autoregressively.
* The latter however proves to be challenging, as generating tokens from nothing in one step is not possible.
  * diffusion solves this, but requires additional steps at best and a separate model at worse, just for one RVQ level.
  * the normal NAR (RVQ level 1+) does not face this problem, as it's already given a sufficient initial sequence of tokens to work with, and thus only requires one step.

The implemented solution follows a similar paradigm to diffusion, but with masking instead of noise.
* incidentally, [this paper](https://arxiv.org/abs/2406.05478) demonstrates this in the use of a NAR transformer for image generation

To-do: fill out this more when it works.

## Embeddings

The "magic" of subjugating a transformer for audio use lies within the ensemble of the embeddings. This is necessary as each piece of a sequence is fundamentally different, but a HF-compatible model can geta way with treating each sequence as separate ranges within a total token sequence.

While embeddings *can* be tied to the output head, testing showed that the model ***really*** does not like to do this, although my implementation could very well be flawed.

### Text Embeddings

The input text phonemes (or output for STT) are passed through an embedding head (`text`), similar to how a normal text LLM would. Nothing fancy is required, as it's very straightforward.

Technically, due to how the audio embeddings are implemented, it's possible to offer "language specific" embeddings, rather than one unified IPA-based embedding + a language embedding (`lang`).
* Such an implementation *could* in fact inference from normal text rather than IPA phonemes.

These embeddings *could* instead be added on top of the input prompt embedding instead of serving as additional tasks (similar to injecting position embeddings), but additional experimentation is required to see if the model both can work under this and/or benefits from this.

#### Language Embedding

This embedding provides the requested language for the model to be aware of.

This *mostly* isn't necessary, but VALL-E X's paper mentions needing a token for the language itself, and other solutions like XTTS2 provides a language token as well.

In practice, this seems to help govern the accent general mannerisms associated with that language. For example, prompting French or German with the language set to `en` will give typical foreigner speech of trying to speak a language they don't know.
* Consequently, since this does tie to accents more, ***extreme*** attention is to be paid to the dialects being trained against, instead of naively grouping, say, all of Spansih to one language code.

This embedding probably helps the model with being able to perform cross-lingual outputs, but I did not do any experimentations on a model without this, as the reference `ar+nar-llama-8` was trained with this from the beginning (and maybe the `ar+nar-retnet-8` experiment).

#### Tone Embedding

This embedding *should* provide information on the tone for the model to output the utterance in.

Should, since I do not actually make use of this anywhere, and the model is not trained against any tones. I would need to annotate my dataset based on tones *and* pick which tones I do want.

This should most definitely help the model identify tone strongly even without needing to annotate for it, but it does an adequate already with maintaining tone from a given input prompt.

### Audio Embeddings

However, due to the nature of the encoded audio, embedding the audio tokens requires the dark arts, as we use audio both as an input prompt (`prom`) for guidance, and as an output response (`resp`).

As EnCodec encodes audio across eight codebooks (and DAC's 44Khz audio under nine codebooks), our audio is encoded under a 2D space, rather than a simple 1D space like text does. Because of this, we require embeddings for *every* codebook level, effectively giving eight embedding heads for audio.
* Technically, this can be stored within a unified embedding head, but each layer is offset by 1024 (the number of tokens).

For the `prom` embedding, we can simply use each embedding for each layer. Each embedding level maps to its respective RVQ level.

Howver, the `resp` requires some extra care, as the model needs to both causally (AR) and parallel-ly (NAR) decode tokens.
* The first embedding level pertains to RVQ level 0 for the AR.
* The remaining embedding levels maps to RVQ level 0 + n for the NAR.
  * In other words, embedding level 1 => RVQ level 0, embedding level 2 => RVQ level 1, etc...
* I believe this is because the model needs to "know" whether to predict ~~the next token in the sequence, or the token in the same position of the next RVQ level~~ which tokens of a given embedding.
  * In other words, the AR's RVQ level 0 embedding predicts itself, while the NAR's embeddings predict the next level's embeddings.
  * Unfortunately, providing a token for the current/target RVQ level within the input sequence doesn't seem to help? I don't remember if I experimented with this or not, but testing of a "sane" `resp` embedding proved to be unfruitful.

The `prom` and `resp` are split since, in theory, it helps the model know better what audio to source from, and what audio is part of the output sequence. In theory.
* I have yet to conduct tests with interchanging the `prom` and `resp`, and the model definitely exhibits being able to map from the `prom` directly, and being able to inference from the `prom` being prefixed in the `resp`.

Finally, the model *may* then sum each embedding level back down to one sequence, as defined under `cfg.model.experimental.audio_embedding_sums`.
* The resulant sum is not normalized by the length.
* It's not a requirement, as the model can still function only "seeing" the required RVQ level.
* However, it *may* help to have the model being able to "see" prior levels, as one RVQ level might depend on the prior level.
  * This is mostly dependent on the underlying audio model being used, which would depend on how each residual is defined.
* A model not trained with summing embeddings can enable it without much impact, but a model trained on summing embeddings cannot go in the other way without further training.
  * It *could* be beneficial to train a model under mixed modes, but requires experimentation.
  * The reference model was trained originally without summing, then trained with summing.

Additionally, it's *technically* possible to instead use the embeddings from the core model used to encode the audio, but in theory this may exclude specific features the model has encoded within the embeddings.

#### RVQ Level Embedding

This embedding hints what the target RVQ level of the audio codes is being targetted. This embedding is not required, but seems some architectures (Mamba) requires this.

This *may* replace needing separate embeddings for each RVQ level, but experimentation is required to test this claim.

### Tasks

The base model handles processing inputs into token sequences, per the requested task assigned to each input in a batch.

Most sequences follow a `<text><RVQ level><language><prompt><output>` sequence, but some tasks will receive the prompt as a list of tensors, instead.

The nitty gritty of how each task is implemented is documented under [./docs/data.md](/docs/data.md).

#### Text-to-Speech

The primary zero-shot text-to-speech synthesis `tts` task takes in a requested text transcript, a piece of reference audio, and then outputs the response audio of the utterance saying the prompted transcript.

The model primarily functions in a zero-shot setting, where it does not need a guiding prefix, but few-shotting is possible through manual intervention.
* I believe the original VALL-E paper refers to this more as `VALL-E Continuous`, while some other TTS solutions follow this method by transcribing the input audio prompt as well.

Additional tasks are implemented in this project, but ***are yet to be trained for*** in the reference model (as some tasks require additional compute-cost).

##### Noise Suppression

This task `ns` aims to suppress or remove noise from the input audio.

In practice, this task is already implemented by providing the input audio to denoise, and having the input transcription be the transcription of the input audio. The output isn't 1:1 exact in terms of prosody and delivery, but it's close.

I imagine training for this task will better help the model understand what is noise and what isn't, and can better strongly-er map utterances from the input audio prompt to use in the output, delivering better prompt adherance.
* This also might help serve in helping the model identify effects applied to an utterance, and being able to maintain it in normal `tts` tasks, such as reverb or the audio quality itself (the "acoustic environment").

##### Speech Removal

This task `sr` aims to remove speech from a given audio, effectively serving as the reverse of denoising.

As state above, this should help the model better identify what is noise and what isn't.

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

#### Speech-to-Text

The speech-To-text `stt` task transcribes a given piece of audio, by taking an input encoded audio, and outputting the text transcription.

However, due to the model being trained on phonemes, the resultant output is the phonemes itself.

The primary benefit of this task is to provide a fast way to directly transcribe audio into the phonemes used annotate the dataset itself, but at the moment the reference model isn't accurate enough to rely on this.
* The other problem is it's very hard to validate this, as the output isn't in English, and requires processing through the model again to verify the transciption.

This task will follow a reverse sequence of `<audio><language><RVQ level><output>`.

## Emergent Behavior

The model can be prompted in creative ways to yield some interesting behaviors:
* prompting without an input audio prompt will have the model generate a random voice at the "cost" of some unintelligible utterance at the beginning of the output response (despite doing no promptless training).
  * finetunes / LoRAs can benefit from this by having input audio promptless synthesis, while opting to have an input audio prompt for guidance.
* prompting with an input text prompt being the transcription of the input audio prompt will have the response follow very closely to the input prompt  (despite not doing input=output training).
  * this should allow for easy transcription editing without much fuss.

# `models/*`

This folder contains scripts relating to models and code for VALL-E use, from the wrapping model to the underlying arch.

## `models/lora.py`

This script implements Low-Ranking Adapters, to allow for cheaper and easier finetuning of existing modules.

At the moment, two approaches are offered, through replacing `nn.Linear` outright, or parameterizing a `nn.Liner`. The latter is used by default(?).

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

## `models/ar_nar.py`

This script implements VALL-E as a unified autoregressive and non-autoregressive model, where RVQ-level 0 is inferenced autoregressively, the remaining levels are infereneced non-autoregressively.

By default, this is the default model, but is used through `cfg.model.capabilities = ["ar", "nar"]`.

For training, this model handles preparing the batch provided through the dataloader according to a randomly sampled targetted RVQ-level.

For inferencing, this will dynamically inference depending on the arguments provided.

## `models/ar.py`

This script implements VALL-E as a pure autoregressive (AR) model.

If `cfg.model.experimental.interleave=True`, this makes use of interleaving its audio codes, instead of inferencing per-codebook level. If not, this simply attends to RVQ level 0.

This model serves as an experiment that failed, and might be revisited in the future.

Use of this is governed through `cfg.model.capabilities = ["ar"]`

## `models/nar.py`

This script implements VALL-E as a mostly-pure non-autoregresive model, where it infers the duration autoregressively (if `"len" in cfg.model.capabilities`). If not, this simply attends to RVQ levels 1+.

This makes use of training an additional `len` task that can infer the duration of a requested input, as well as (maybe) using special tokens as the initial input for RVQ-level 0 (the level the AR attends to).

This model serves as an experiment that failed, and might be revisited in the future.

Use of this is governed through `cfg.model.capabilities = ["nar"]`

## `models/experimental.py`

This script implements VALL-E as a mostly-HuggingFace compatible model, where it handles processing tokens as a uniform sequence of IDs.

This mostly serves as an experiment to see what is required to do so, for possible future implementations requiring just `llama.cpp` and `encodec.cpp`, and to provide a pure HF-compatible implementation.

Use of this is governed through `cfg.model.experimental.hf = True`

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

### `models/arch/bitnet.py`

This script modifies modules of BitNet to play nicely with my existing code.

### `models/arch/llama.py`

This script modifes modules of LLaMA provided through `transformers`.

A bulk of it pertains to modifying `LlamaAttention` and detecting available attention mechanisms, allowing for using different attention mechanisms:
* `torch.nn.functional.scaled_dot_product_attention`-based attention:
  * `math`: torch's SDPA's `math` kernel
  * `mem_efficient`: torch's SDPA's memory efficient (`xformers` adjacent) kernel
  * `cudnn`: torch's SDPA's `cudnn` kernel
  * `flash`: torch's SDPA's flash attention kernel
* internal implementations of external attention backends:
  * `xformers`: [facebookresearch/xformers](https://github.com/facebookresearch/xformers/)'s memory efficient attention
  * `flash_attn`: uses the available `flash_attn` package (including `flash_attn==1.0.9` through a funny wrapper)
  * `flash_attn_v100`: uses [ZRayZzz/flash-attention-v100](https://github.com/ZRayZzz/flash-attention-v100/)'s Flash Attention for Volta (but doesn't work currently)
  * `fused_attn`: uses an implementation using `triton` (tested on my 7900XTX and V100s), but seems to introduce errors when used to train after a while
  * `default`: uses the naive path for hte internal implementation (used for attention-debugging purposed)
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

### `models/arch/mamba.py`

This script modifies modules of Mamba, to allow it to play nicely with my existing code.

If I rememer right, it just simply provides gradient checkpointing.

### `models/arch/mixtral.py`

Like `llama.py`, this provides modifications to Mixtral through `transformers`.

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

### `models/arch/mamba_vasqu`

This folder contains an implementation of Mamba2 as a HuggingFace-compatible model, and not requiring Triton.

Attributions are noted at the top of the respective file(s).

### `models/arch/retnet_syncdoth`

This folder contains scripts to modify modules within a RetNet model.

Attributions are noted at the top of the respective file(s).