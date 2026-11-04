# Training

`python -m vall_e.train --store <root> --out training/<run> ...` (`vall_e/train.py`); `scripts/train.sh <out> <args>`
wraps it with resume-on-crash. `scripts/recipe.sh` has the flagship's stages with their exact settings.

## The recipe

| Stage | Steps | What | Result |
|---|---|---|---|
| `base` | 300k from scratch | masked generation, cosine mask ratios, text + prompt dropped together 10% (CFG), filtered store, lr 3e-4, WSD | reads text (text use 5.4%) |
| `prompts` | 30k | 3-10 s prompts | long prompts stop looping |
| `more` | 300k | prompt-only dropout 10% added, lr 1e-4 (+ the recovered Emilia/DE) | WER 7.4%, SIM 0.722 with speaker guidance |
| `length` | 6k | a length head on the frozen model | predicted lengths within ~15% |

225M parameters, batch 16 (up to 16k positions), bf16, AdamW (0.9 / 0.95, weight decay 0.1), gradient clipping 1.0,
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
