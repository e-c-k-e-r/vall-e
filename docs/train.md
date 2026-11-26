# Training

`python -m vall_e.train --store <root> --out training/<run> ...` (`vall_e/train.py`); `scripts/train.sh <out> <args>`
wraps it with resume-on-crash. `scripts/v4-chain.sh` has v2.0's stages with their exact settings, `scripts/recipe.sh`
the 225M's.

## v2.0 (746M)

| Stage | Steps | What |
|---|---|---|
| `base` | 300k from scratch | as the 225M's, lr 2e-4; voiced game dialogue and Genshin (`--extra-store`) at 15% of draws; held-out speakers (`data/holdout-speakers-v2.json`) never trained on; the text-use gate at 30k |
| `prompts` | 30k | 3-10 s prompts |
| `more` | 300k | prompt-only dropout 10%, targets up to 30 s at the store's natural share of long lines |
| `length` | 6k | a length head on the frozen model |

1536 wide, 24 layers, 24 heads, activation checkpointing (`--grad-ckpt`; without it 518M already ran out of 24 GB):
1.1-1.25 s per step on an RX 7900 XTX, peaking at 19 GB, about 8 days in all. Before it, both stores were
re-phonemized (`python -m vall_e.store rephonemize`): they had been phonemized by an older espeak (15% of English lines
differed from what inference produces) and without the abbreviation handling.

Then the release steps:

* **Averaging** (`scripts/avg.py`, `scripts/v4-polish.sh`): the mean of `more`'s 100k, 200k and 300k checkpoints, then a
  new length head: UTMOS +0.12 to +0.30 over the last checkpoint, SIM level or better. This is v2.0. (300k more steps
  instead lost to it, `scripts/v4-more2.sh`.)
* **Guidance distillation** (`scripts/v4-distill.sh`): v2.0 as the teacher at guidance 3.0 / 2.0, 20k steps at lr 2e-5:
  v2.0-fast.
* **Length heads** (`scripts/len-promptless.sh`): every released head trained on 3k more steps with 20% of prompts
  dropped, so promptless lengths are right.

## v1.x (225M)

v1.0 is the recipe below (`v3-main6-len`); v1.1 averaged its last checkpoints; v1.2 is 0.7 of a long-form fine-tune
mixed with 0.3 of v1.1, and v1.3 a Genshin fine-tune mixed half and half with v1.2 (`scripts/mix.py`, WiSE-FT weight
mixing; `scripts/genshin-chain.sh`), each with a new length head, the last one promptless-aware.

| Stage | Steps | What | Result |
|---|---|---|---|
| `base` | 300k from scratch | masked generation, cosine mask ratios, text + prompt dropped together 10% (CFG), filtered store, lr 3e-4, WSD | reads text (text use 5.4%) |
| `prompts` | 30k | 3-10 s prompts | long prompts stop looping |
| `more` | 300k | prompt-only dropout 10% added, lr 1e-4 (+ the recovered Emilia/DE) | WER 7.4%, SIM 0.722 with speaker guidance |
| `length` | 6k | a length head on the frozen model | predicted lengths within ~15% |

Batch 16 (up to 16k positions), bf16, AdamW (0.9 / 0.95, weight decay 0.1), gradient clipping 1.0,
`torch.compile` per block: about 0.32 s per step on an RX 7900 XTX, about 2.5 days in all.

## What was learned getting there

The first full-store runs learned the audio but barely read the text, and their teacher-forced losses matched a model
that did. In order of how much it mattered:

1. **Too many objectives from the start.** Every addition trained in from scratch (length head, accent embedding,
   3-10 s prompts, similar prompts, noisy prompts, speaker balancing, independent dropout, AR rows) cost a little text
   reading, and together they stopped it (text use 4.6% -> 1.2%). The text accounts for only ~5% of the flow loss, so
   it is easily crowded out. So: train text reading first, add tasks after, as fine-tunes or frozen heads.
2. **Fixed 0.8 mask ratio from scratch**: never trains generation without audio anchors.
3. **Overrunning LibriVox / Audiobooks segments** (`docs/data.md`): taught babbling past the end of the text.
4. **CFG's null condition barely trained** when text and prompt were only dropped independently (1% both).

## Watching a run

* **`val/text_use`**, logged at every eval: the relative loss increase when each validation item gets another item's
  text. A run that reads text reaches ~5% (at batch 32, within ~8k steps; at batch 16 about twice as many); the broken
  runs sat at 0-1%. `scripts/textuse-gate.sh <run> <step> <min>` stops a run below the bar. Scale the step with the
  batch size.
* **Sampled WER**, not teacher-forced loss: `scripts/evaluate.sh` on a checkpoint.
* **WSD plateaus look worse than they are.** Mid-run checkpoints of `more` scored below the model it started from
  (SIM -0.03 to -0.05); a 10k-step decay to zero from step 100k (a "decay probe") already beat it. Judge a WSD run at
  its end or with a probe.

## Options

Data: `--groups`, `--languages`, `--min-rate`, `--prompt-s`, `--similar-p`, `--noisy-p`, `--speaker-alpha`,
`--pad-p` (silence-padded targets), `--max-tokens`. Objectives: `--mask-ratio` (`cos` or a fixed ratio), `--cond-drop`
/ `--text-drop` / `--prompt-drop`, `--ar-frac`, `--len-frac`, `--accent`, `--ctc-weight`. Model: `--d-model`,
`--layers`, `--heads`, `--patch`, `--head`. Schedule: `--lr`, `--warmup`, `--steps`, `--decay-frac`, `--init-from`,
`--train-only`, `--keep-every`.

## Machine notes (the ROCm box)

Train with torch 2.13 (`$VALL_E_TRAIN_PY`); torch 2.7 segfaults in MIOpen. Occasional segfaults still happen;
`scripts/train.sh` resumes from the last checkpoint (every 2k steps). Runs get a private `TMPDIR`, as something cleaning
`/tmp` once deleted the DataLoader's sockets. Don't run CPU inference or scoring next to a training run: two OOMs took
the whole session down.
