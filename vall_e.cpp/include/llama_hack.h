#pragma once

#include "llama-vocab.h"
#include <array>

/* Begin cringe so I can access the model's tok_embd */
// it needs to be copied so the struct layout is exactly as it is under llama.cpp
#define LLAMA_MAX_LAYERS  512
#define LLAMA_MAX_EXPERTS 160  // DeepSeekV2

enum e_model {
	MODEL_UNKNOWN,
};

enum llm_arch {
	LLM_ARCH_UNKNOWN,
};

struct llama_hparams_posnet {
	uint32_t n_embd;
	uint32_t n_layer;
};

struct llama_hparams_convnext {
	uint32_t n_embd;
	uint32_t n_layer;
};

struct llama_hparams {
	bool vocab_only;
	bool rope_finetuned;
	bool use_par_res;
	bool swin_norm;

	uint32_t n_vocab = 0;
	uint32_t n_ctx_train; // context size the model was trained on
	uint32_t n_embd;
	uint32_t n_embd_features = 0;
	uint32_t n_layer;
	uint32_t n_rot;
	uint32_t n_swa = 0; // sliding window attention (SWA)
	uint32_t n_embd_head_k; // dimension of keys (d_k). d_q is assumed to be the same, but there are n_head q heads, and only n_head_kv k-v heads
	uint32_t n_embd_head_v; // dimension of values (d_v) aka n_embd_head
	uint32_t n_expert = 0;
	uint32_t n_expert_used = 0;
	uint32_t n_vocab_type = 0; // for BERT-style token types
	uint32_t n_rel_attn_bkts = 0;

	// for WavTokenizer
	struct llama_hparams_posnet   posnet;
	struct llama_hparams_convnext convnext;

	std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_arr;
	std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_kv_arr;
	std::array<uint32_t, LLAMA_MAX_LAYERS> n_ff_arr;

	uint32_t n_layer_dense_lead = 0;
	uint32_t n_lora_q = 0;
	uint32_t n_lora_kv = 0;
	uint32_t n_ff_exp = 0;
	uint32_t n_ff_shexp = 0;
	uint32_t n_expert_shared = 0;
	float    expert_weights_scale = 0.0;

	float f_norm_eps;
	float f_norm_rms_eps;
	float f_norm_group_eps;

	uint32_t n_norm_groups;

	float f_attn_logit_softcapping = 50.0f;
	float f_final_logit_softcapping = 30.0f;

	// for RWKV
	uint32_t rescale_every_n_layers = 0;
	uint32_t time_mix_extra_dim = 0;
	uint32_t time_decay_extra_dim = 0;
	uint32_t wkv_head_size = 0;

	float     rope_attn_factor = 1.0f;
	float     rope_freq_base_train;
	float     rope_freq_scale_train;
	uint32_t  n_ctx_orig_yarn;
	float     rope_yarn_log_mul;
	int       rope_sections[4];

	// for State Space Models
	uint32_t ssm_d_conv  = 0;
	uint32_t ssm_d_inner = 0;
	uint32_t ssm_d_state = 0;
	uint32_t ssm_dt_rank = 0;
	bool ssm_dt_b_c_rms = false;

	float f_clamp_kqv      = 0.0f;
	float f_max_alibi_bias = 0.0f;
	float f_logit_scale    = 0.0f;

	// Additional scale factors (Granite/Granite MoE)
	float f_residual_scale  = 0.0f;
	float f_embedding_scale = 0.0f;
	float f_attention_scale = 0.0f;

	bool causal_attn   = true;
	bool use_alibi     = false;
	bool attn_soft_cap = false;

	// needed by encoder-decoder models (e.g. T5, FLAN-T5)
	// ref: https://github.com/ggerganov/llama.cpp/pull/8141
	llama_token dec_start_token_id = LLAMA_TOKEN_NULL;

	enum llama_pooling_type      pooling_type            = LLAMA_POOLING_TYPE_NONE;
	enum llama_rope_type         rope_type               = LLAMA_ROPE_TYPE_NONE;
	enum llama_rope_scaling_type rope_scaling_type_train = LLAMA_ROPE_SCALING_TYPE_NONE;
};

struct llama_model {
	e_model     type  = MODEL_UNKNOWN;
	llm_arch    arch  = LLM_ARCH_UNKNOWN;
	llama_ftype ftype = LLAMA_FTYPE_ALL_F32;

	std::string name = "n/a";

	llama_hparams hparams = {};
	llama_vocab   vocab;

	struct ggml_tensor * tok_embd = nullptr;
    struct ggml_tensor * type_embd = nullptr;
    struct ggml_tensor * pos_embd = nullptr;
    struct ggml_tensor * tok_norm = nullptr;
    struct ggml_tensor * tok_norm_b = nullptr;

    struct ggml_tensor * output_norm = nullptr;
    struct ggml_tensor * output_norm_b = nullptr;
    struct ggml_tensor * output = nullptr;
    struct ggml_tensor * output_b = nullptr;
    struct ggml_tensor * output_norm_enc = nullptr;

    // classifier
    struct ggml_tensor * cls = nullptr;
    struct ggml_tensor * cls_b = nullptr;
    struct ggml_tensor * cls_out   = nullptr;
    struct ggml_tensor * cls_out_b = nullptr;

    struct ggml_tensor * conv1d = nullptr;
    struct ggml_tensor * conv1d_b = nullptr;
};

/* BEGIN VALL-E SPECIFIC HELPERS */
struct ggml_tensor * llama_get_embedding_weights(struct llama_model * model) {
    return model->tok_embd;
}
struct ggml_tensor * llama_get_output_head_tensor(struct llama_model * model ) {
    return model->output;
}
void llama_set_output_head(struct llama_model * model, struct ggml_tensor* tensor ) {
    // set the output tensor
    model->output = tensor;
    // required to properly output logits
    *const_cast<uint32_t*>(&model->hparams.n_vocab) = tensor->ne[1];
}
/* END VALL-E SPECIFIC HELPERS */

/* End cringe code */