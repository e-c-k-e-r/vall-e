# Model Notes

To be filled.

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