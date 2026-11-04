# The model

`vall_e/model.py`. A transformer reads `[text][prompt][target]` and a flow-matching head generates the target audio as
continuous NeMo FSQ values; masked generation (NAR-demasking) fills a target of known length in about 25 passes.

## Audio representation

`nvidia/audio-codec-44khz` quantizes each 86.13 Hz frame as 8 groups of 4 scalars at `[8, 5, 5, 5]` levels (1000 codes
per group), and its decoder only sees those scalars as values in [-1, 1] (`vall_e/fsq.py` converts codes, per-scalar
level indices and values).

* **Values, not tokens.** Earlier versions treated each group as an unordered 1000-way token, with one classifier per
  group predicted independently: an off-by-one scalar cost as much as noise, and nothing tied a frame's groups
  together. v3 reads and writes the values themselves.
* **Patches.** Two frames form one position: 64 values, 43 positions per second.
* **Input:** one linear projection of a patch's values; masked positions get a learned mask vector.
* **Output: a flow-matching head** (`FlowHead`, a small AdaLN MLP, MAR-style) generates a position's 64 values jointly
  from its hidden state by rectified flow (8 Euler steps on F5-TTS's sway grid), then every value snaps to its grid. A
  per-scalar classifier head (`head="classifier"`) is still in the code: it was intelligible but strained (harmonics
  above ~1 kHz broke up where it was unsure, as each scalar was picked independently), which is what the flow head
  fixed: WER 18.7% -> 12.3%, SIM 0.36 -> 0.53 at equal size.

## Backbone

RMSNorm pre-norm, SwiGLU, rotary embeddings, plus QK-norm and a sigmoid gate on the attention output (as recent LLMs
use: bounded activations, no attention sinks). The flagship is 225M parameters: 1024 wide, 16 layers, 16 heads.

Segment and mode embeddings mark text / prompt / target positions and the mode (NAR-demask, AR, LEN).

## Generation (`demask_flow`)

The target starts fully masked; over 25 steps, positions are revealed in random order on a cosine schedule, each
sampled by the flow head from the current hidden states. Guidance:

* **CFG** (`cfg`, default 6.0): the velocity is pushed away from an unconditional pass with text and prompt dropped
  (trained with both dropped together 10% of the time). 6.0 beat 2.5 on both WER and SIM.
* **Speaker guidance** (`cfg_prompt`, default 2.0 in `tts.py`): a third pass with the text but no prompt, pushing
  towards the voice specifically. Needs a model trained with prompt-only dropout (`--prompt-drop`). +0.022 SIM at 2.0
  on the flagship; at 3.0 SIM rises further but German and Chinese WER start to climb.
* Lower flow temperature and longer prompts than 6 s did not help (see `docs/eval.md`).

## Length

The target length is an input. A length head (`length=True`: a query position after text + prompt regresses the log
length) predicts it; on the flagship it is trained on the frozen backbone (`--train-only len_q length`), so it cannot
affect generation. Predicted lengths are within ~15% of the true ones and cost nothing against reference lengths.

## AR mode (optional)

The same weights can generate left to right (`ar=True`): a prefix-LM mask, a start embedding, the previous patch as
history for the flow head, a stop logit, head-only CFG through a learned null condition, and a KV-cached decoder
(`generate_ar`, `KVCache`). Mixed into training from the start, AR rows learned to continue the audio rather than read
the text; a bolt-on AR fine-tune of an aligned model got about half its samples right. Deprioritised: the flagship is
NAR only.

## Not adopted (tried or considered)

* **Fixed 0.8 mask ratio from scratch** (AutoNAT): the model never had to generate without audio anchors and never
  learned to read the text. Cosine-sampled ratios cover every ratio the demasking loop visits.
* **CTC phoneme-recognition auxiliary loss**: halved text use; over unmasked positions it teaches recognition from the
  audio, the opposite of reading the text.
* **Silence-padded targets** against over-long lengths: robust to 1.4x lengths, but worse at normal ones.
* **Hybrid linear attention, MTP, MoE, sparse attention**: not needed at this size and sequence length (30 s is about
  1300 positions). `fla-core` 0.5.2 does train Gated DeltaNet / KDA on ROCm if streaming AR comes back.

## Literature that shaped it

* DiTAR (2502.03930): head-only CFG and the previous patch as AR history; MELLE (2407.08551): weighted stop logit.
* MAR (2406.11838), SALAD (2410.16048): continuous heads, random reveal order beats confidence order.
* NVIDIA's frame-stacking study (2509.19592): independent prediction of stacked FSQ frames collapses SIM, joint
  prediction recovers it, which matches the classifier vs flow result.
* MaskGCT (2409.00750): WER flat past ~25 demask steps, SIM saturates by ~10, as here.
* F5-TTS (2410.06885): sway sampling for few flow steps.
