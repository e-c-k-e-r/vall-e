# What is VALL-E?

[VALL-E](https://arxiv.org/abs/2301.02111) describes how treating text-to-speech synthesis as a language problem can easily be solved with a language model. The original paper utilizes a basic transformer as the underlying architecture to perform zero-shot text-to-speech synthesis using a short audio prompt as reference.

# Why VALL-E?

At the time, state-of-the-art neural-based TTS solutions were sparing. TorToiSe had a similar approach to treating TTS as a language problem, but required a ton of additional cruft on top of its ensemble. Thus, when VALL-E's paper released, it was simple yet effective with it requiring, at the time, just an AR and a NAR model, and leaving EnCodec to handle the rest (feature extraction, encoding audio, decoding audio). Vocos then improves upon EnCodec's decoding to produce better quality audio.

# Why this VALL-E?

Unlike the paper, this VALL-E aims to:
* be lightweight as possible, only requiring one model to load and use (and EnCodec/Vocos as an audio encoder/decoder).
	+ Even the original VALL-E requires a separate AR and a NAR.
* keep training and finetuning (be it the base model or through LoRAs) accessible to anyone.
	+ Bark was needlessly complex in providing even additional voices to use.
	+ Current SoTA such as F5-TTS supports it, but seems to have a rather high ceiling to finetune it. 
* provide decent zero-shot text-to-speech synthesis, both without requiring sampling adjustments and providing thorough sampler settings.
* provide additional, easy to use functionality, that other solutions don't offer.

However, at this point and time, the implementation is rather divorced from VALL-E and its derivating papers, but the core principle is still followed.

## To-Do

* [x] train and release a serviceable model for finetuning against.
* [x] train and release a ***good*** zero-shot model.
  - for what it's worth it's decent enough for me to finally be happy with it.
* [ ] well-integrated training through the Web UI (without the kludge from ai-voice-cloning)
* [x] ~~explore alternative setups, like a NAR-only model or Descript-Audio-Codec~~
  - the current experiment of an AR length-predictor + NAR for the rest seems to fall apart...
  - Descript-Audio-Codec 44KHz has NAR issues, but this *might* be user error.
* [x] ~~explore better sampling techniques~~
  - the AR doesn't *need* exotic sampling techniques, as they're bandaids for a bad AR.
  - the NAR benefits from greedy sampling, and anything else just harms output quality.
* [x] clean up the README, and document, document, document.
* [x] extend to multiple languages ([VALL-E X](https://arxiv.org/abs/2303.03926)).
  - reference model is trained against English, Japanese, French, and German.
* [ ] extend to addditional tasks ([SpeechX](https://arxiv.org/abs/2308.06873)).
  - `stt` (Speech-to-Text) seems to be working fine for the most part.
  - other tasks seem to require a ton of VRAM......
* [ ] extend using [VALL-E 2](https://arxiv.org/pdf/2406.05370)'s features (grouped code modeling + repetition aware sampling)
  - desu these don't seem to be worthwhile improvements, as inferencing is already rather fast, and RAS is just a fancy sampler.
* [ ] audio streaming
  - this *technically* can work without any additional architecture changes, just clever tricks with sampling-then-decoding-to-audio.
  - something similar to HiFiGAN (or the one for TorToiSe) trained on the last hidden states of the AR *might* also enable an alternate way for streaming.
* [ ] speed up inferencing for the AR
  - KV caching both yields broken output and quadratically slow output, unless I'm doing something grossly wrong.
  * [x] provide a pure NAR model that foregoes most of the inferencing slowdowns a regular AR+NAR model will provide.
* [ ] HF-ify the model
  * [x] write a weights converter
  * [ ] implement a pure llama_HF implementation
  - this might be easily possible by subjugating the tokenizer to handle all the embeddings / classifiers
  - this will pave the way to use the model under an easy marriage of `llama.cpp` and `encodec.cpp`
* [ ] replace the phonemizer with something that doesn't depend on espeak
  * [ ] train the model to handle text => phoneme (without a hit to the rest of the model)
    * [ ] ...and phonemes => text
    * [ ] allow raw text as input instead
  - espeak is nice, but I can only really put my whole trust with phonemizing English.
  - a small model trained to handle converting text to phonemes might work, but has it's own problems (another model to carry around, as accurate as the dataset it was trained against, requires training for each language... etc).
* [ ] smarter/clever inferencing, such as:
  * [x] "rolling" context, where the last generated sentence is the prefix for the next sentence.
* [ ] explore exotic features like:
  * using a pure text vocab rather than IPA phonemes (as a transformer should be "smart" enough to map text tokens)
  * interleaving by using summed embedding tokens:
    * for example, `<RVQ 0-7><RVQ 0>` => `<RVQ 0-7><RVQ 0-1>` => `<RVQ 0-7><RVQ 0-2>` (etc.)
    * however, I imagine the sequences to train for this are *too* exotic.
  * mixing multiple speakers through summing input prompt embeddings
    * I do not expect this to work, but you never know...

## "Postmortem"

For the most part, the model is complete. With the `NAR-len` being crammed on, I'm satisifed with the performance-to-quality.

However, while this solution boasts being lightweight, there are some caveats for its given size
* its at capacity on what it *can* do without additional tasks to augment it further
  * post-fixing it with additional layers glued on doesn't seem to offer very much improvement (12 => 16 layers)
* wrangling it is a bit of a chore, as some voices work fine under the `AR` but not the `NAR-len`, and vice-versa
  * some voices outright refuse to work without LoRA training
  * some sampler settings works on some voices, but others need some tweaking
* for short durations, it excels, but despite training on longer durations, stability is less guaranteed
* subjugating an existing LLM architecture is a bit of a pain, as I would *love* to make full use of LLaMA niceties
  * `hf`-ifying it is possible, but it'd be a chore to set up the tokenizer properly
* it still seems like the phase of the moon matters with how it wants to cooperate
  * some eval tests it seems fine, other times issues like word errors will crop up
* the `NAR-len` requires CFGs > 2-ish to cooperate (or a prefix)
  * this isn't *so* much of an issue, but this can lead to user error, and CFG incurs an additional sampling step per step.
  * guidance distillation would be nice, but distillation in general harms finetuning (assuming this just as likely harms it)
  * rolling context/prefix does solve this
    * VALL-E Continuous (prefixing with the input prompt) could also fix this, but technically makes it one-shot and not zero-shot


## Notices and Citations

Unless otherwise credited/noted in this repo or within the designated Python file, this repository is [licensed](/LICENSE) under AGPLv3.

- [EnCodec](https://github.com/facebookresearch/encodec) is licensed under CC-BY-NC 4.0. If you use the code to generate audio quantization or perform decoding, it is important to adhere to the terms of their license.

- This implementation was originally based on [enhuiz/vall-e](https://github.com/enhuiz/vall-e), but has been heavily, heavily modified over time. Without it I would not have had a good basis to muck around and learn.

```bibtex
@article{wang2023neural,
  title={Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers},
  author={Wang, Chengyi and Chen, Sanyuan and Wu, Yu and Zhang, Ziqiang and Zhou, Long and Liu, Shujie and Chen, Zhuo and Liu, Yanqing and Wang, Huaming and Li, Jinyu and others},
  journal={arXiv preprint arXiv:2301.02111},
  year={2023}
}
```

```bibtex
@article{defossez2022highfi,
  title={High Fidelity Neural Audio Compression},
  author={Défossez, Alexandre and Copet, Jade and Synnaeve, Gabriel and Adi, Yossi},
  journal={arXiv preprint arXiv:2210.13438},
  year={2022}
}
```
