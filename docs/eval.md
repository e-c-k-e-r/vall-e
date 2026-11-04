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

Earlier checks with 6-16 items disagreed with each other by more than the differences being judged; at 237 paired
items, differences of 1-2 WER points and 0.01 SIM are resolvable.

```
python -m vall_e.evalset run --set evalsets/v3-eval.json --ckpt gt --out training/evals/gt
python -m vall_e.evalset run --set evalsets/v3-eval.json --ckpt <ckpt> --out <dir> --length predict --cfg-prompt 2.0
python -m vall_e.evalset report <dir> ... --gt training/evals/gt
python -m vall_e.evalset compare <dir A> <dir B> --gt training/evals/gt
```

A run takes about 6 minutes on the 7900 XTX (generation ~4, scoring ~2).

## Results

Predicted lengths, guidance 6.0 unless noted.

| Model | WER | SIM | en WER / SIM |
|---|---|---|---|
| ground truth (ceiling) | 7.3% | 0.717 | 6.6% / 0.703 |
| `v3-main5p-len`, guidance 2.5 | 10.7% | 0.656 | 5.8% / 0.602 |
| `v3-main5p-len` | 8.1% | 0.685 | 5.1% / 0.652 |
| `v3-main6-len` | 6.9% | 0.700 | 4.8% / 0.661 |
| **`v3-main6-len`, speaker guidance 2.0** (the flagship) | **7.4%** | **0.722** | **3.9% / 0.686** |
| `v3-main6-len`, speaker guidance 3.0 | 8.2% | 0.730 | 4.1% / 0.695 |

Flagship per language (speaker guidance 2.0): en 3.9% / 0.686, de 5.7% / 0.760, fr 7.5% / 0.728, ja 11.7% / 0.756,
ko 5.1% / 0.753, zh 12.7% / 0.728.

SIM slightly past the "ceiling" means the output resembles the prompt clip a little more than the speaker's other
recordings do (partly by copying the prompt's recording conditions); SIM-target (0.648) shows the remaining distance
to the speaker's actual recording of the sentence.

Paired results that settled defaults (`v3-main5p-len`): guidance 6.0 vs 2.5: SIM +0.028, WER -2.6. Speaker guidance
needs prompt-only dropout in training. 10 s prompts vs 6 s: SIM unchanged, WER +2.9. 3 s: SIM -0.065. Flow temperature
0.8 / 0.6: SIM -0.011 / -0.029. Predicted vs reference length: no cost. Length slack (1.2x): WER +3.0. Silence-padded
training: robust to 1.4x lengths, but +1.6 WER at normal lengths.

## Machine note

Build the Whisper pipeline before importing anything heavy: with the old `vall_e.emb.similar` imported first, every
transcription took ~30 s instead of ~0.2 s on this ROCm setup. `evalset.py` transcribes everything first, then scores
SIM, and loads models from the local cache (`HF_HUB_OFFLINE=1`) so a network blip can't fail a run.
