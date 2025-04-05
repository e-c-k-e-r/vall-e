# vall_e.cpp

This is an implementation that makes use of [llama.cpp](https://github.com/ggerganov/llama.cpp/) and [encodec.cpp](https://github.com/PABannier/encodec.cpp).

Model weights can:
* be found at [`ecker/vall-e@gguf`](https://huggingface.co/ecker/vall-e/tree/gguf)
* converted with `vall_e.export --yaml=./model_path/config.yaml --hf`, then running `python3 /path/to/your/llama.cpp/convert_hf_to_gguf ./model_path/hf/`

## Build

Populate `./include/` with the `ggml`, `llama.cpp`, and `encodec.cpp` headers.

Populate `./lib/` with the compiled libraries of `llama.cpp`, `encodec.cpp`, and `espeak-ng` (if not already in your `LD_LIBRARY_PATH`).

Run `make`.

### Required Modifications

[`encodec.cpp`](https://github.com/PABannier/encodec.cpp) requires updating its GGML copy to the latest version, which requires a few lines to get the CPU backend working (per my [fork](https://github.com/e-c-k-e-r/encodec.cpp)).

[`llama.cpp`](https://github.com/ggerganov/llama.cpp) only possible modification needs to ensure that a non-causal attention mask is used; everything necessary can be hacked together with clever tricks.
* initially written on commit `9ba399dfa7f115effc63d48e6860a94c9faa31b2`, updated to commit `7a84777f42a9b3ba47db5d20b7662f8ddf92f652`

## To-Do

* [x] converted model to GGUF
	* [x] convert it without modifying any of the existing code, as the tokenizer requires some care
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
	* [ ] proper scoring
* [x] working `NAR` output
	* [x] `NAR` sampling
* [x] decode audio to disk
* [x] a functional CLI
* [x] actually make it work
* [x] clean up to make the code usable elsewhere
* [x] configured to allow for being used as a lib
	* (I do need to validate this in my engine project, but that's in MSYS2)
* [ ] feature parity with the PyTorch version
	* [ ] vocos
	* [ ] additional tasks
		* [ ] `stt`
		* [x] `ns` / `sr`
		* [ ] samplers