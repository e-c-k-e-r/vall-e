# vall_e.cpp

This is an implementation that makes use of [llama.cpp](https://github.com/ggerganov/llama.cpp/) and [encodec.cpp](https://github.com/PABannier/encodec.cpp).

At the moment it's ***very*** barebones as I try and wrestle with `llama.cpp`'s API without needing to modify its code.

## Build

Populate `./include/` with the `llama.cpp` and `encodec.cpp` headers.

Populate `./libs/` with the compiled libraries of `llama.cpp` and `encodec.cpp`.

Run `make`.

### Required Modifications

[`encodec.cpp`](https://github.com/e-c-k-e-r/encodec.cpp) requires updating its GGML copy to the latest version, which requires a few lines to get the CPU backend working.

[`llama.cpp`](https://github.com/e-c-k-e-r/llama.cpp) *might* not require any modifications, but implementing `LLM_ARCH_VALL_E` requires some surgery.

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
* [ ] working `AR` output
	* [x] `AR` sampling
	* currently need a model that didn't regress with the `AR:0:0` output
* [ ] working `NAR-len` output
	* [x] `NAR-len` sampling
	* need to assert that a non-causal mask is used
* [ ] working `NAR` output
	* [x] `NAR` sampling
	* need to assert that a non-causal mask is used
* [x] decode audio to disk
* [ ] a functional CLI
* [ ] actually make it work
