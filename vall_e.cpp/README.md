# vall_e.cpp

This is an implementation that makes use of [llama.cpp](https://github.com/ggerganov/llama.cpp/) and [encodec.cpp](https://github.com/PABannier/encodec.cpp).

At the moment it's ***very*** barebones as I try and wrestle with `llama.cpp`'s API without needing to modify its code.

## Build

Populate `./include/` with the `llama.cpp` and `encodec.cpp` headers.

Populate `./libs/` with the compiled libraries of `llama.cpp` and `encodec.cpp`.
* `encodec.cpp` requires updating `ggml` to the latest version and doing a quick hack to make it work on the CPU backend.
* `llama.cpp` currently requires no hacks, but would be *very* nice to hack in a way to retrieve a model's `tok_embd`.

Run `make`.

## To-Do

* [x] converted model to GGUF
	* [ ] convert it without modifying any of the existing code, as the tokenizer requires some care
* [x] basic framework
	* [x] load the quantized model
	* [x] orchestrate the required embeddings
	* [x] juggle the output head / classifier properly
* [ ] phonemize text
	* with the help of espeak-ng
* [ ] tokenize phonemes
	* the tokenizer is being a huge thorn on actual sequences
* [x] load audio from disk
* [x] encode audio
* [x] sum embeddings for the `prom` and prior `resp`s
* [x] `AR` sampling
* [ ] `NAR-len` demasking sampling
* [x] `NAR` sampling
* [x] decode audio to disk
* [ ] a functional CLI
* [ ] actually make it work
	* it seems naively stitching the model together isn't good enough since the output is wrong, it most likely needs training with a glued together classifier