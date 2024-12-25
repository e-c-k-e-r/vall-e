# vall_e.cpp

This is an implementation that makes use of [llama.cpp](https://github.com/ggerganov/llama.cpp/) and [encodec.cpp](https://github.com/PABannier/encodec.cpp).

At the moment it's ***very*** work in progress.

Model weights can be found at [`ecker/vall-e@gguf`](https://huggingface.co/ecker/vall-e/tree/gguf).

## Build

Populate `./include/` with the `ggml`, `llama.cpp`, and `encodec.cpp` headers.

Populate `./libs/` with the compiled libraries of `llama.cpp`, `encodec.cpp`, and `espeak-ng`.

Run `make`.

### Required Modifications

[`encodec.cpp`](https://github.com/PABannier/encodec.cpp) requires updating its GGML copy to the latest version, which requires a few lines to get the CPU backend working (per my [fork](https://github.com/e-c-k-e-r/encodec.cpp)).

[`llama.cpp`](https://github.com/ggerganov/llama.cpp) only possible modification needs to ensure that a non-causal attention mask is used; everything necessary can be hacked together with clever tricks.

## To-Do

* [x] converted model to GGUF
	* [ ] convert it without modifying any of the existing code, as the tokenizer requires some care
* [x] basic framework
	* [x] load the quantized model
	* [x] orchestrate the required embeddings
	* [x] juggle the output head / classifier properly
* [x] phonemize text
	* with the help of espeak-ng
* [x] tokenize phonemes
	* tokenize with `llama_tokenize` instead of a homebrewed method because the tokenizer is being a huge thorn
* [x] load audio from disk
* [x] encode audio
* [x] sum embeddings for the `prom` and prior `resp`s
* [x] working `AR` output
	* [x] `AR` sampling
* [x] working `NAR-len` output
	* [x] `NAR-len` sampling
* [x] working `NAR` output
	* [x] `NAR` sampling
* [x] decode audio to disk
* [ ] a functional CLI
* [x] actually make it work
* [ ] clean up to make the code usable
* [ ] feature parity with the PyTorch version
	* [ ] vocos
	* [ ] additional tasks (`stt`, `ns`, `sr`, samplers)