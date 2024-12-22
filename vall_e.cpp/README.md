# vall_e.cpp

This is an implementation that makes use of [llama.cpp](https://github.com/ggerganov/llama.cpp/) and [encodec.cpp](https://github.com/PABannier/encodec.cpp).

At the moment it's ***very*** barebones as I try and wrestle with `llama.cpp`'s API without needing to modify its code.

## Build

Populate `./include/` with the `llama.cpp` and `encodec.cpp` headers.

Populate `./libs/` with the compiled libraries of `llama.cpp` and `encodec.cpp`.

Run `make`.


### Required Modifications

`encodec.cpp` requires updating its GGML copy to the latest version, which requires a few lines to get the CPU backend working.
`llama.cpp` *might* not require any modifications, but:
* `llm.build_vall_e` can mostly copy `llm.build_llama`, but with:
	* `KQ_mask = build_inp_KQ_mask( lctx.cparams.causal_attn )`
	* a unified output head (pain)
		* OR adjusting the `model.output` to the correct classifier head
	    * OR slicing that tensor with the right range (`ggml_view_2d` confuses me)
		* both require also require `*const_cast<uint32_t*>(&ctx->model.hparams.n_vocab) = output->ne[1];` because the logits are tied to `n_vocab`
* commenting out `GGML_ABORT("input/output layer tensor %s used with a layer number", tn.str().c_str());` because grabbing embeddings/classifiers require using `bid` to trick it thinking it's part of a layer
* some helper functions to retrieve the embeddings tensor from the model
* some helper functions to set the target classifier head

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