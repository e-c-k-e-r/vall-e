# Data

`vall_e/store.py`, `vall_e/data.py`, `vall_e/noisy.py`.

## The store

One canonical, memory-mapped copy of the encoded audio and its index (no per-speaker JSON at training time, no HDF5
duplicate):

```
<root>/manifest.json         codec, shard list, and the tables ids index into: groups, speakers (and their group),
                             languages, phoneme vocabulary
<root>/<shard>/codes.u16     every utterance's NeMo codes, concatenated, raw uint16 [frames, 8]
<root>/<shard>/*.npy         per utterance: frame offsets, speaker, language, phoneme ids and offsets, text, names
```

Shards are append-only: new data is new shards. Readers keep only small global arrays in memory and memory-map the
rest, so DataLoader workers share pages. `Store(root)` checks every shard's codes against its index on open.

Sidecars, built after conversion and loaded if present:

| File | What | Built by |
|---|---|---|
| `similar.npy` | each utterance's 8 most similar same-speaker utterances (from the metadata's `similar` lists) | `store.py similar` |
| `exclude.npy` / `.json` | utterances dropped from training, and why | `store.py exclude` |
| `noise/` | the noise clips (no transcripts) | `noisy.py import-noise` |
| `noisy/` | noised copies of one utterance per speaker (60k speakers), for noisy prompts | `noisy.py bake` |
| `silence.npy` | NeMo's codes for digital silence (one constant frame), for `--pad-p` | encoded once from 3 s of zeros |

The current store: 10.11M utterances, 200k speakers, about 16.7k hours, 28 groups, languages en / fr / de / ja / zh / ko,
106 phonemes; 42 shards, about 80 GB.

## Building it

`python -m vall_e.store convert --meta <metadata dir> --data <.nem dir> --out <root>` converts the old per-speaker
metadata JSON + `.nem` artifacts (`--resume` continues an interrupted run; `--supplement` adds what a store is missing,
which recovered 220k Emilia/DE utterances split across `Emilia/DE/<speaker>` and `Emilia/DE/Emilia-DE/<speaker>`).
A tool that goes straight from raw audio to the store (VAD, ASR with word timestamps, cutting at aligned pauses, a
second-ASR consistency check, encoding) is still to be written; the old preparation scripts are in the `v2-final` tag.

## Known data problems

* **LibriVox and Audiobooks segments overrun their transcripts**: about a quarter run into the next sentence ("I have
  found David my servant" + "...with my help"), ground-truth WER 10-13% against LibriTTS-R's 2.7%. They are 74% of the
  store. Overrunning segments have too few phonemes for their length, so training drops LibriVox / Audiobooks items
  under 13 phonemes per second (`--min-rate 13 --min-rate-groups LibriVox Audiobooks`): LibriVox keeps 70% at 2.5% WER,
  Audiobooks 52% at 8.6%. 7.05M utterances remain. Re-cutting them from word-aligned transcripts would recover the rest.
* **Mislabelled transcripts** (`exclude.json`, 2,532 utterances): kanji that espeak read as "Japanese letter", Whisper
  hearing screams in English-voiced games as Japanese, and `housamo` items phonemized as English.
* **Near-duplicate prompts**: 19% of Emilia/JA's most-similar utterances are the same line again; `StoreUtterances`
  skips similar-prompt candidates whose normalised text shares the first and last 8 characters with the target.

## Training items (`StoreUtterances`, `Collator`, `BucketSampler`)

* Speakers split train / val by a hash of the speaker key (1% val), so a speaker is never in both.
* A prompt is another utterance of the same speaker, cropped once per batch to a length drawn from `--prompt-s` (the
  flagship: 3-10 s); optionally from the similar lists (`--similar-p`) or a noised prompt (`--noisy-p`).
* Batches are length-bucketed and capped at `--max-tokens` positions (rows x the longest row, counting the longest
  prompt), split into equal parts: zh / ja phoneme strings run up to ~40 phonemes per second, and a fixed batch of
  long rows ran out of memory.
* `--speaker-alpha` evens out speakers *within* each group (balancing across the whole store gave Emilia 83% of draws).
