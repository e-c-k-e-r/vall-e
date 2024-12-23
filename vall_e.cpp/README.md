# vall_e.cpp

This is an implementation that makes use of [llama.cpp](https://github.com/ggerganov/llama.cpp/) and [encodec.cpp](https://github.com/PABannier/encodec.cpp).

At the moment it's ***very*** barebones as I try and wrestle with `llama.cpp`'s API without needing to modify its code.

## Build

Populate `./include/` with the `llama.cpp` and `encodec.cpp` headers.

Populate `./libs/` with the compiled libraries of `llama.cpp` and `encodec.cpp`.

Run `make`.

### Required Modifications

[`encodec.cpp`](https://github.com/e-c-k-e-r/encodec.cpp) requires updating its GGML copy to the latest version, which requires a few lines to get the CPU backend working.
[`llama.cpp`](https://github.com/e-c-k-e-r/llama.cpp) *might* not require any modifications, but:
* `llm.build_vall_e` can mostly copy `llm.build_llama`, but with:
	* `KQ_mask = build_inp_KQ_mask( lctx.cparams.causal_attn )`
	* a unified output head (pain)
		* OR adjusting the `model.output` to the correct classifier head (better option)
	    * OR slicing that tensor with the right range (`ggml_view_2d` confuses me)
		* both require also require `*const_cast<uint32_t*>(&ctx->model.hparams.n_vocab) = output->ne[1];` because the logits are tied to `n_vocab`
* commenting out `GGML_ABORT("input/output layer tensor %s used with a layer number", tn.str().c_str());` because grabbing embeddings/classifiers require using `bid` to trick it thinking it's part of a layer
* some helper functions to retrieve the embeddings tensor from the model
* some helper functions to set the target classifier head
* some fix for `GGML_ASSERT(mask->ne[0] == a->ne[0])` when using a non-causal attention mask (or I can test on the model that had a causal NAR......)

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
	* [ ] `NAR-len` sampling
	* currently cannot inference with non-causal_attn
* [ ] working `NAR` output
	* [x] `NAR` sampling
	* currently cannot inference with non-causal_attn
* [x] decode audio to disk
* [ ] a functional CLI
* [ ] actually make it work
