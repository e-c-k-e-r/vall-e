# What is VALL-E?

[VALL-E](https://arxiv.org/abs/2301.02111) describes how treating text-to-speech synthesis as a language problem can easily be solved with a language model. The original paper utilizes a basic transformer as the underlying architecture to perform zero-shot text-to-speech synthesis using a short audio prompt as reference.

# Why VALL-E?

At the time, state-of-the-art neural-based TTS solutions were sparing. TorToiSe had a similar approach to treating TTS as a language problem, but required a ton of additional cruft on top of its ensemble. Thus, when VALL-E's paper released, it was simple yet effective with it requiring, at the time, just an AR and a NAR model, and leaving EnCodec to handle the rest (feature extraction, encoding audio, decoding audio). Vocos then improves upon EnCodec's decoding to produce better quality audio.

# Why this VALL-E?

Unlike the paper, this VALL-E aims to:
* be lightweight as possible, only requiring one model to load and use (and EnCodec/Vocos as an audio encoder/decoder).
	+ Even the original VALL-E requires two separate models (one for the course codes, and one for the fine codes).
* keep training and finetuning (be it the base model or through LoRAs) accessible to anyone.
	+ Bark was needlessly complex in providing even additional voices to use.
	+ Current SoTA such as F5-TTS supports it, but seems to have a rather high ceiling to finetune it. 
* provide decent zero-shot text-to-speech synthesis, both without requiring sampling adjustments and providing thorough sampler settings.
* provide additional, easy to use functionality, that other solutions don't offer.

However, at this point and time, the implementation is *very* divorced from VALL-E and its derivating papers, but the core principle is still followed.

# Why *not* this VALL-E?

This VALL-E is still actively being iterated upon without any actual proper standards or procedures.
* While I try to maintain interop with previous versions, I can't guarantee it (for example, support for `ar+nar-retnet-8` dropped due to shifting focuses).
* I am *very* stubborn with/against some approaches, paradigms, and methodologies.

There are far better TTS solutions out there, such as [MaskGCT](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct) and [F5-TTS](https://github.com/SWivid/F5-TTS). They're both easy to use and offer amazing results.

In the future, a 44KHz model will be released if training goes well for it.

## Model Specifications

The reference model (`ar+nar-llama-8`/`ar+nar-len-llama-8`):
* boasts 220M parameters
* supports English, German, French, Japanese, Korean, and Chinese (Mandarin?)
* has several modalities of inferencing:
  * the primary audio level (RVQ level 0) can be inferenced both autoregressively (`AR`) or non-autoregressively (`NAR-len`)
    * pure-NAR can yield faster-than-realtime output
  * supports predicting the duration of an input
  * supports Speech-to-Text (although it's a second-class feature)
  * supports additional tasks such as speech removal, noice reduction, and voice converison.
    * additional tasks such as speaker extraction and speech editing eventually™ (just need to train on it)
* trained on `?` samples / `?` hours of EnCodec-quantized audio at 24KHz

## To-Do

* [x] train and release a serviceable model for finetuning against.
* [x] train and release a ***good*** zero-shot model.
  - for what it's worth it's decent enough for me to finally be happy with it.
* [ ] train a serviceable model for 44KHz audio (instead of 24KHz)
* [ ] well-integrated training through the Web UI (without the kludge from ai-voice-cloning)
* [x] clean up the README, and document, document, document.
  * [ ] cleanup the documentation again, as most of it feels like schizorambling......
* [x] extend to multiple languages ([VALL-E X](https://arxiv.org/abs/2303.03926)).
  - reference model is trained against English, Japanese, French, German, Korean, and Chinese (Mandarin?).
  - [x] improve multi-lingual support
  - [ ] improve cross-lingual support
* [ ] extend to addditional tasks ([SpeechX](https://arxiv.org/abs/2308.06873)).
  - `stt` (Speech-to-Text) seems to be working fine for the most part, but is very much a second-class feature.
  - other tasks seem to require a ton of VRAM......
  - SpeechX tasks might need to be reworked to fit well within the `NAR-len` context to make full use of masking (for example, for speech editing)
  - ***possibly*** voice conversion through the `NAR-len` with clever demasking tricks (for example, the tokens that are masked are from the source voice)
* [ ] audio streaming
  - this *technically* can work without any additional architecture changes, just clever tricks with sampling-then-decoding-to-audio.
  - something similar to HiFiGAN (or the one for TorToiSe) trained on the last hidden states of the AR *might* also enable an alternate way for streaming.
  - desu the `NAR-len` can be fast enough with short enough utterances to generate audio >1x speeds
* [ ] speed up inferencing for the AR
  - KV caching both yields broken output and quadratically slow output, unless I'm doing something grossly wrong.
  * [x] provide a pure NAR model that foregoes most of the inferencing slowdowns a regular AR+NAR model will provide.
* [x] HF-ify the model
  * [x] write a weights converter
  * [x] implement a pure llama_HF implementation
    * provided under `./vall_e/models/base.py`'s `__main__`
* [ ] replace the phonemizer with something that doesn't depend on espeak
  * [ ] train the model to handle text => phoneme (without a hit to the rest of the model)
    * [ ] ...and phonemes => text
    * [ ] using a pure text vocab rather than IPA phonemes (as a transformer should be "smart" enough to map text tokens)
    * these features are predicated on the model being trained for it
* [ ] smarter/clever inferencing, such as:
  * [x] inference *all* codebooks in one pass, rather than each level being its own discrete pass.
      * `cfg.model.version >= 7` models will rely on this
      * these features are predicated on the model being trained for it
  * [x] "rolling" context, where the last generated sentence is the prefix for the next sentence.
  * [ ] for the AR, stop inferencing sequences in the batch that has already hit its stop token
* [x] objective metrics such as WER / SIM-O
  * [x] WER simply requires transcribing audio then computing word error rates through the transcriptions
  * [x] SIM-O requires passing the raw waveform through a speaker-similarity model
* [x] valle.cpp through llama.cpp + encodec.cpp
  * extend to decode with vocos.cpp, instead, for a quality improvement
* [ ] 44KHz audio, through either DAC or `nvidia/audio-codec-44khz`
  * the former has quality issues in the higher RVQ levels, but may be resolved with the experimental implementation
  * the latter needs testing, as it being an FSQ codec requires extra care

## "Postmortem"

For the most part, the model is complete. With the `NAR-len` being crammed on, I'm satisifed with the performance-to-quality.

However, while this solution boasts being lightweight, there are some caveats for its given size
* its at capacity on what it *can* do without additional tasks to augment it further
  * post-fixing it with additional layers glued on doesn't seem to offer very much improvement (12 => 16 layers)
  * the only bet is to feed it more data and see how it fares, since the model is still grossly undertrained compared to the 50K+ hour behemoths.
* subjugating an existing LLM architecture is a bit of a pain, as I would *love* to make full use of LLaMA niceties
  * `hf`-ifying it is possible, but due to the nature of summed audio embeddings and split classifiers, it's not as plug-and-play as I would like for inferencing.
* speaker similarity is rather mediocre for unseen speakers, the model isn't as robust for mapping speakers to its latent space as it is for seen speakers.
* despite being rather robust, some vocal stutters makes it way in.

### "Postmortem" ""Postmortem""

The model even working at all might entirely be a fluke.

A naive embedding implementation (`./vall_e/models/base.py`) manages to "just work" for EnCodec, while other audio codecs (DAC, `nvidia/audio-codec-44khz`) fail to converge meaningfully.

A more codec-aware embedding/classifier implementation (`./vall_e/models/base_v2.py`) fails to properly learn all levels for any codec, even with all the additional cruft to help things. Even scaling the model up just has the gradients seem a little more chaotic with about the same training progression.
* However it seems just giving it time will have things eventually sort itself out, maybe.

## Notices and Citations

Unless otherwise credited/noted in this repo or within the designated Python file, this repository is [licensed](/LICENSE) under AGPLv3.

- [EnCodec](https://github.com/facebookresearch/encodec) is licensed under CC-BY-NC 4.0. If you use the code to generate audio quantization or perform decoding, it is important to adhere to the terms of their license.

- This implementation was originally based on [enhuiz/vall-e](https://github.com/enhuiz/vall-e), but has been heavily, heavily modified over time. Without it, I would not have had a good basis to muck around and learn.

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

```bibtex
@inproceedings{emilia,
    author={He, Haorui and Shang, Zengqiang and Wang, Chaoren and Li, Xuyuan and Gu, Yicheng and Hua, Hua and Liu, Liwei and Yang, Chen and Li, Jiaqi and Shi, Peiyang and Wang, Yuancheng and Chen, Kai and Zhang, Pengyuan and Wu, Zhizheng},
    title={Emilia: An Extensive, Multilingual, and Diverse Speech Dataset for Large-Scale Speech Generation},
    booktitle={Proc.~of SLT},
    year={2024}
}
```

```bibtex
@INPROCEEDINGS{librilight,
  author={J. {Kahn} and M. {Rivière} and W. {Zheng} and E. {Kharitonov} and Q. {Xu} and P. E. {Mazaré} and J. {Karadayi} and V. {Liptchinsky} and R. {Collobert} and C. {Fuegen} and T. {Likhomanenko} and G. {Synnaeve} and A. {Joulin} and A. {Mohamed} and E. {Dupoux}},
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Libri-Light: A Benchmark for ASR with Limited or No Supervision}, 
  year={2020},
  pages={7669-7673},
  note = {\url{https://github.com/facebookresearch/libri-light}},
}
```