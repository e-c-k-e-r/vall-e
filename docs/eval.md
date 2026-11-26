# Evaluation

`vall_e/evalset.py`, `scripts/evaluate.sh`.

## The eval set

`evalsets/v3-eval.json`: 237 frozen items from validation speakers (never trained on), 182 speakers. English is 87 items
stratified over LibriTTS-R, LibriVox, Audiobooks and Emilia/EN; 30 each of French, German, Japanese, Chinese, Korean
(Emilia). Each item has a fixed prompt: another utterance of the same speaker, a fixed 6 s crop.

* **WER / CER**: Whisper large-v3-turbo in the item's language; words for en / fr / de, characters for zh / ja / ko;
  pooled per language. A ground-truth run (`--ckpt gt`, the real codes decoded) is the ceiling, and its items scoring
  over 30% (24 of 237: bad transcripts, Whisper hallucinating) are left out of reports and comparisons.
* **SIM**: WavLM-large + ECAPA-TDNN speaker similarity against the item's prompt clip (SIM-O style; the ground truth
  scores 0.717, the ceiling) and against the speaker's real recording of the sentence (SIM-target).
* **Intervals**: 95% bootstrap intervals per language; `compare` gives a paired bootstrap of the difference between
  two runs on the same items, with errors capped at 100% per item so one runaway loop can't outweigh the rest.

* **UTMOS**: predicted naturalness (UTMOS22 strong, the VoiceMOS 2022 winner), 1-5; trained on English, so compare it
  within a language.

Earlier checks with 6-16 items disagreed with each other by more than the differences being judged; at 237 paired
items, differences of 1-2 WER points and 0.01 SIM are resolvable. Runs are deterministic (a fixed seed per item): the
same model and settings give the same scores.

Four more sets, built the same way (`evalset build`), for what the main set doesn't cover:

| Set | Items | What |
|---|---|---|
| `v3-long.json` | 41 | long multi-sentence lines (en / de), 10-30 s targets |
| `v3-games-holdout.json` | 52 | game voices held out of training (`data/holdout-speakers-v2.json`), en / ja |
| `v3-genshin-holdout.json` | 60 | Genshin characters held out of training, en / ja |
| `v3-nikke.json` | 60 | Nikke characters, never trained on (an eval-only store), en, 3 s prompts |

```
python -m vall_e.evalset run --set evalsets/v3-eval.json --ckpt gt --out training/evals/gt
python -m vall_e.evalset run --set evalsets/v3-eval.json --ckpt <ckpt> --out <dir> --length predict --cfg-prompt 2.0
python -m vall_e.evalset report <dir> ... --gt training/evals/gt
python -m vall_e.evalset compare <dir A> <dir B> --gt training/evals/gt
```

A run takes about 6 minutes on the 7900 XTX (generation ~4, scoring ~2).

## Results

The released models (`README.md`), predicted lengths, each at its own guidance (v2.0 3.0 / 2.0, v2.0-fast none, v1.3
6.0 / 2.0). WER / CER, SIM, UTMOS; items with bad references left out, as above.

| Model | main | en | long | held-out games | Genshin | Nikke |
|---|---|---|---|---|---|---|
| ground truth | 7.3% / 0.717 / 3.14 | 6.6% / 0.703 / 3.72 | 4.7% / 0.786 / 3.29 | 3.1% / 0.567 / 2.68 | 5.2% / 0.618 / 2.91 | 1.2% / 0.492 / 2.73 |
| **v2.0** | **7.1% / 0.744 / 3.42** | 5.1% / 0.709 / 4.08 | 5.7% / 0.733 / 3.82 | 6.3% / 0.611 / 3.46 | 5.5% / 0.689 / 3.37 | 2.7% / 0.538 / 3.28 |
| v2.0-fast | 7.2% / 0.733 / 3.50 | 5.1% / 0.698 / 4.09 | 3.8% / 0.713 / 3.89 | 5.0% / 0.582 / 3.57 | 5.9% / 0.682 / 3.45 | 2.3% / 0.527 / 3.25 |
| v1.3 | 6.0% / 0.739 / 3.27 | 3.7% / 0.705 / 3.97 | 8.3% / 0.703 / 3.68 | 6.3% / 0.587 / 3.32 | 6.5% / 0.675 / 3.21 | 2.5% / 0.513 / 3.17 |

Per language on the main set (WER / CER, SIM):

| | en | de | fr | ja | ko | zh |
|---|---|---|---|---|---|---|
| ground truth | 6.6% / 0.703 | 6.1% / 0.753 | 6.8% / 0.738 | 6.6% / 0.733 | 8.0% / 0.743 | 10.1% / 0.650 |
| v2.0 | 5.1% / 0.709 | 4.1% / 0.787 | 7.2% / 0.742 | 9.9% / 0.775 | 6.6% / 0.777 | 10.1% / 0.747 |
| v2.0-fast | 5.1% / 0.698 | 5.9% / 0.772 | 7.2% / 0.740 | 10.9% / 0.775 | 5.1% / 0.756 | 10.5% / 0.728 |
| v1.3 | 3.7% / 0.705 | 5.7% / 0.783 | 6.3% / 0.730 | 7.0% / 0.775 | 3.6% / 0.767 | 12.0% / 0.744 |

v2.0 has the highest similarity, most clearly on voices outside the training data (held-out games +0.024 SIM and Nikke
+0.025 over v1.3), and reads long lines better (WER 5.7% against 8.3%); v1.3 reads slightly more accurately on the main
set. The distilled v2.0-fast trades 0.01-0.03 SIM for the highest UTMOS of the three, at twice v2.0's speed. The held-out game voices' Japanese
(8 items of acted, shouted lines) is where all three are weakest.

SIM past the "ceiling" means the output resembles the prompt clip more than the speaker's other recordings do (partly
by copying the prompt's recording conditions); SIM-target (about 0.65 on the main set for all three) shows the remaining
distance to the speaker's actual recording of the sentence.

Without a prompt (`--prompt-seconds 0`, a random voice), the length heads trained with 20% of prompts dropped (the
released ones) cut the over-long predictions that the model filled with repeated words: main set WER v1.3 9.1% ->
6.5%, v2.0 English 7.9% -> 5.4% (the earlier heads predicted promptless lengths 1.3-1.85x too long; now 1.12x, as with
a prompt).

Speaker guidance on v2.0 (guidance 3.0): 3.0 / 4.0 against 2.0 raise SIM (+0.022 / +0.042 on Nikke, +0.008 / +0.036 on
the held-out games) but lower UTMOS (-0.07 to -0.18) and add ~1 WER point; 2.0 stays the default.

### Development history (225M, `v3-main6-len`)

Predicted lengths, guidance 6.0 unless noted; the first release (v1.0) was `v3-main6-len` with speaker guidance 2.0.

| Model | WER | SIM | en WER / SIM |
|---|---|---|---|
| ground truth (ceiling) | 7.3% | 0.717 | 6.6% / 0.703 |
| `v3-main5p-len`, guidance 2.5 | 10.7% | 0.656 | 5.8% / 0.602 |
| `v3-main5p-len` | 8.1% | 0.685 | 5.1% / 0.652 |
| `v3-main6-len` | 6.9% | 0.700 | 4.8% / 0.661 |
| `v3-main6-len`, speaker guidance 2.0 (v1.0) | 7.4% | 0.722 | 3.9% / 0.686 |
| `v3-main6-len`, speaker guidance 3.0 | 8.2% | 0.730 | 4.1% / 0.695 |

Paired results that settled defaults (`v3-main5p-len`): guidance 6.0 vs 2.5: SIM +0.028, WER -2.6. Speaker guidance
needs prompt-only dropout in training. 10 s prompts vs 6 s: SIM unchanged, WER +2.9. 3 s: SIM -0.065. Flow temperature
0.8 / 0.6: SIM -0.011 / -0.029. Predicted vs reference length: no cost. Length slack (1.2x): WER +3.0. Silence-padded
training: robust to 1.4x lengths, but +1.6 WER at normal lengths.

## Machine note

Build the Whisper pipeline before importing anything heavy: with the old `vall_e.emb.similar` imported first, every
transcription took ~30 s instead of ~0.2 s on this ROCm setup. `evalset.py` transcribes everything first, then scores
SIM, and loads models from the local cache (`HF_HUB_OFFLINE=1`) so a network blip can't fail a run.
