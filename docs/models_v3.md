# Model V3 Notes (draft)

This is a design draft for a `_v3` class of models. Nothing here is trained yet; sections marked *untested* are proposals to verify with the prototype in `vall_e/v3/`.

The goals:
* full 44KHz audio through `nvidia/audio-codec-44khz`, without the capacity plateau the `_v2` models hit
* one model for every modality the EnCodec `ar+nar-len` model serves: NAR-demasking, AR (for streaming, where the inference step maps to real time), duration prediction, and the extra tasks
* a backbone that picks up what recent LLMs settled on, where it pays off at a few hundred million parameters on one GPU

## Why `_v2` plateaus on NeMo

The codec's own config (`model_config.yaml` inside `audio-codec-44khz.nemo`) quantizes each frame with 8 groups of 4 scalars, at `[8, 5, 5, 5]` levels each (8 * 5 * 5 * 5 = 1000 codes per group). A frame is therefore a 32-dim vector of ordered values, and the decoder only ever sees those values (`nemo.py`, `FiniteScalarQuantizer.decode`).

`_v2` treats every group as an unordered 1000-way token:
* an embedding table and a classifier per group, so predicting one scalar off by one step is as wrong as predicting noise
* all 8 groups are predicted independently from one hidden state per frame, so nothing ties a frame's groups together; with an RVQ codec the levels at least refine a shared coarse prediction, with FSQ they are peers
* 86 frames per second makes sequences long for a small model

The first two are properties of how the model reads the codec, not of the codec.

## Audio representation

* **Values, not tokens**: each frame is dequantized to its 32 values in [-1, 1] (`(digit - L // 2) / (L // 2)` per scalar, as the codec does). Masked frames use a learned mask vector instead.
* **Patching**: 2 frames form one model position (64 values, 43 positions per second). Odd lengths are padded with a repeat of the last frame and trimmed after decoding. Patch size is a config value; 4 (21.5 positions per second) is worth an ablation.
* **Input**: one linear projection of the patch (64 -> `d_model`), replacing the per-group embedding tables and the `AudioEncoder` MLP.
* **Output, stage 1 (classifier head)**: one small classifier per scalar (8- or 5-way), 64 per position, as one `d_model -> 368` projection split per scalar. The target is soft: most mass on the true level and a little on its neighbours, so near misses cost less. This keeps per-token confidence scores, so the existing demasking loop works unchanged.
* **Output, stage 2 (flow head)** *(untested)*: a small MLP conditioned on the hidden state generates the whole 64-value patch by flow matching, then every value snaps to its grid. This models the joint distribution within a patch, which the classifier head cannot. Demasking then picks positions in random order, as in MAR, since there is no token probability to rank by.

Both heads decode with the unchanged NeMo decoder, so no codec training is needed.

## Modalities in one model

The EnCodec `ar+nar-len` model shows one set of weights can serve AR, NAR and NAR-demasking. `_v3` keeps that, with the mode given by:
* a learned mode embedding added to every audio position (`ar`, `nar-demask`, `len`), standing in for the separate AR/NAR embeddings of the older models
* the attention mask: causal for AR, full for NAR-demasking

| Modality | Attention | Target | Stop condition |
|---|---|---|---|
| NAR-demask | full | masked patches | length from the `len` task |
| AR | causal | next patch | a stop logit on every position *(untested)* |
| len | full | one position, multi-digit classifier (as in `_v2`) | n/a |

Training mixes modes per sample (to start: 60% NAR-demask, 30% AR, 10% len).

### Hybrid linear attention and MTP *(untested)*

Streaming AR wants constant memory per step and cheap long prompts, which is what the hybrid stacks in recent LLMs give: Gated DeltaNet (or KDA) layers with a full attention layer every 4th (Qwen3.8, GLM-5.3). The problem is that a linear-attention layer is a causal recurrence, while NAR-demasking needs every position to see the future. Two options, to be compared:
* **causal everywhere except the attention layers**: in NAR mode the recurrent layers still run causally, and only the full attention layers (1 in 4) see the whole sequence. No dead weights, but `_v2` found an all-causal mask cripples NAR-demasking, so this may be too little bidirectional context.
* **bidirectional scan in NAR mode**: the recurrent layers run forward and backward over the same weights and sum the two, as bidirectional state-space vision models do. The weights serve both modes; whether one set of weights can learn both directions is the open question.

An MTP head (one extra block predicting the patch after next from the last hidden state and the next patch's embedding, as in DeepSeek-V3 and Qwen3.8) adds a training signal and self-speculative decoding for AR. In NAR mode it is unused, but it is one block.

Neither is in the first prototype: it uses full attention only, to settle whether the representation works before adding AR machinery.

## Backbone

Adopted from recent LLMs, all cheap at this scale:
* **QK-norm**: RMSNorm on queries and keys per head. The `_v2` notes report the gradient norm growing under bfloat16, which this targets.
* **Gated attention output**: a per-head sigmoid gate on the attention output (Qwen3.8), which suppresses attention sinks and keeps activations bounded.
* **Partial rotary embeddings**: rotary on a fraction of each head's dimensions (Qwen3.8 uses 25%). Config value; whether it helps duration generalization is to be measured.
* RMSNorm pre-norm and SwiGLU, as `_v2`.

Considered and left out:
* **MoE**: pays off at parameter counts and data volumes beyond one 7900XTX.
* **Sparse attention (DSA, QSA)**: helps past a few thousand positions; 30 seconds of patched audio is about 1300.
* **Hyper-connections / gated residual streams** (GLM-5.3, Qwen3.8-Flash-Next): reported gains are small at this scale and the change is intrusive. Worth a try once everything else works.

## Training

From the reviews of `_v2` and the `ar+nar-len` recipe:
* mask ratio sampled per sample as `cos(u * pi / 2)`, so training covers every ratio the demasking loop visits
* losses averaged over predicted positions only, then over the batch
* warmup, a long constant phase, then a linear decay to export (WSD); resumable without knowing the total length
* AdamW (betas 0.9 / 0.95, weight decay 0.1 on matrices) in the prototype; Muon (`vall_e/utils/ext/muon.py`) is untried
* validation: per-scalar accuracy, group-exact accuracy (all 4 scalars of a group right) and loss at fixed mask ratios during training; WER / SIM-O, pitch and voicing statistics, and a repair test on decoded samples after (`vall_e/v3/sample.py`, `vall_e/v3/repair.py`). Group-exact accuracy turned out not to track audio quality (see below), so the decoded-audio measures decide
* length-bucketed batches with shuffled batch order, assembled in the DataLoader workers, and `torch.compile` per block: 0.38 to 0.18 s/step at 41M parameters, 0.51 to 0.28 s/step at 225M

## Prototype results

All on held-out speakers (LibriTTS-R and 20k Emilia-EN speakers, 474k utterances), 32 samples, 25 demask steps, CFG 2.5. Decoding the real codes scores WER 5.2% and SIM-O 0.62, the ceiling.

| Model | WER | SIM-O | Median F0 (real 160 Hz) | Voiced frames (real 73%) | Spectral centroid (real 931 Hz) |
|---|---|---|---|---|---|
| `v3-proto`: classifier head, 41M, 50k steps | 18.7% | 0.36 | 227 Hz | 37% | 1313 Hz |
| `v3-flow`: flow head, 49M, 50k steps | 12.3% | 0.53 | 153 Hz | 71% | 1001 Hz |

* The classifier head picks each scalar independently. Where the model is uncertain, a position's 64 values mix incompatible choices: the harmonics above about 1 kHz broke into blobs, and the voice sounded strained. The flow head samples a position's values jointly and fixed this.
* A repair test (mask 15% of a real utterance, fill it in one pass) gave the same per-scalar accuracy for a 225M classifier model as for the 41M one: capacity was not the bottleneck, the head was.
* Overfitting 64 utterances reaches 100% group-exact accuracy, and their generations score the same WER as decoding the real codes, so the representation, demasking loop and decoder path are sound.
* Demask steps (flow model, 8 samples): WER 20% at 8 steps, 18% at 16, 8% at 25, 6% at 50; SIM-O flat at 0.43 to 0.48. Speaker identity needs few steps, intelligibility about 25.

## Literature notes

Findings that shaped the plan below (checked against the papers; numbers as the papers report them).

* **AR flow head needs history.** DiTAR (arXiv 2502.03930), the closest design to v3's AR mode, found WER rose to 22.9% and samples failed to stop when its head saw no previous patch. FELLE (2502.11128) centres the flow's starting noise on the previous frame instead.
* **CFG on the head only.** DiTAR drops the backbone output to a learned null embedding 10% of the time and mixes the head's two velocities, so the backbone runs once per step. Guidance-free training (2501.15420) removes CFG at inference for about 19% more training, and matched CFG on MAR-B, a masked model with a continuous head.
* **Flow-head steps.** DiTAR's WER / SIM barely change between 2 and 10 head steps; F5-TTS (2410.06885) loses little from 32 to 16 with its sway time grid; EPSS (2505.19931) reaches 7 steps without retraining.
* **Annealed CFG and temperature over demask steps.** MAGNeT (2401.04577) anneals CFG 10 to 1 with the mask rate (FAD 0.61 vs 0.63 constant) and temperature 3 to 0; MaskGCT (2409.00750) anneals temperature 1.5 to 0.
* **Demask step floor.** MaskGCT's WER is 8.1 at 5 steps and 2.35 at 25, then flat; SIM saturates by about 10 steps. v3's curve has the same shape, so cheaper steps beat fewer steps.
* **Training mask ratio.** `cos(u * pi / 2)` is SoundStorm's (2305.09636) and Muse's (2301.00704) distribution. AutoNAT (2406.05478) found a fixed 0.8 about equal (FID 8.31 vs 8.40, images, few steps); MAR (2406.11838) and E2 / F5-TTS train at 70-100% masked. No speech paper compares fixed and sampled ratios directly; `vall_e/v3` has an ablation (`--mask-ratio`). MDLM (2406.07524) suggests stratifying the draws across a batch.
* **Reveal order.** For continuous heads, SALAD (2410.16048) found random order beat confidence order; Halton ordering (2503.17076) beat confidence order on images, more so at more steps (`--order vdc` is the 1-D analogue).
* **Joint prediction within a frame stack.** NVIDIA's frame-stacking study (2509.19592) found independent parallel prediction of stacked FSQ frames collapses speaker similarity (0.77 to 0.55 at 4x), and iterative local heads recover part of it, which matches the classifier vs flow result above and argues for 2 frames per position.
* **AR stop.** DiTAR, FELLE and MELLE (2407.08551) use a binary stop logit on the backbone; MELLE weights the single positive frame 100x.
* **AR and NAR in one model.** CosyVoice 2 (2412.10117) serves non-causal, causal and chunked modes from one flow model by sampling the attention mask per example; ARDiT (2406.05551) saw its fully NAR setting less intelligible than its AR ones. No paper found combines masked NAR and continuous AR in one set of weights.
* **Linear attention for TTS.** Lina-Speech (2410.23320) found gated linear attention better than self-attention in the same model (WER 6.9% vs 7.8%, same SIM). Bidirectional use for NAR: alternate the scan direction per layer, or run both directions (a text-model result, untested in TTS).
* `flash-linear-attention` (`fla-core` 0.5.2) trains Gated DeltaNet and KDA on ROCm (RX 7900XTX, Triton 3.7.1): bf16 outputs and gradients within 0.5-0.7% of the fp32 reference.

## Plan

1. ~~NAR-demask, classifier head~~: intelligible but strained (above).
2. ~~Flow head~~: adopted.
3. **Scaled flow model** (1024 dim, 16 layers, 225M): `training/v3-flow-large`.
4. **AR mode** in the same weights: per-sample attention mask (causal or full), next-patch flow target with the previous patch as head input, a weighted stop logit, head-only CFG; KV-cached streaming sampler and chunked NeMo decoding.
5. **Hybrid linear attention and MTP** for AR, comparing causal-only and alternating-direction linear layers for NAR.
6. **Integration** into the framework (`base_v3.py`, `ar_nar_v3.py`, configs, inference), with the duration task and extra tasks.
