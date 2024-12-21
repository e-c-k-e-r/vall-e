# vall_e.cpp

This is an implementation that makes use of [llama.cpp](https://github.com/ggerganov/llama.cpp/) and [encodec.cpp](https://github.com/PABannier/encodec.cpp).

At the moment it's ***very*** barebones as I try and wrestle with `llama.cpp`'s API without needing to modify its code.

## Build

Probably something like:

`g++ -I/path/to/llama.cpp/include/ -L/path/to/llama.cpp/libllama.so -lggml  -lggml-base -lllama -o ./vall_e`

## To-Do

* [x] converted model to GGUF
	* [ ] convert it without modifying any of the existing code
* [x] basic framework
	* [x] load the quantized model
	* [x] orchestrate the required embeddings
	* [x] juggle the output head / classifier properly
* [ ] phonemize text
* [ ] tokenize phonemes
* [ ] load audio from disk
* [ ] encode audio
* [ ] sum embeddings for the `prom` and prior `resp`s
* [x] `AR` sampling
* [ ] `NAR-len` demasking sampling
* [ ] `NAR` sampling
* [ ] decode audio to disk
* [ ] a functional CLI