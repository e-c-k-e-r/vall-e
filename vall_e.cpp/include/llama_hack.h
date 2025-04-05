#pragma once

#include "llama-vocab.h"
#include <array>

/* Begin cringe so I can access the model's tok_embd */
// it needs to be copied so the struct layout is exactly as it is under llama.cpp
#define LLAMA_MAX_LAYERS  512
#define LLAMA_MAX_EXPERTS 160  // DeepSeekV2

enum llm_type {
    LLM_TYPE_UNKNOWN,
};

enum llm_arch {
	LLM_ARCH_UNKNOWN,
};

enum llama_expert_gating_func_type {
    LLAMA_EXPERT_GATING_FUNC_TYPE_NONE    = 0,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX = 1,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID = 2,
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

    uint32_t n_ctx_train; // context size the model was trained on
    uint32_t n_embd;
    uint32_t n_embd_features = 0;
    uint32_t n_layer;
    uint32_t n_rot;
    uint32_t n_swa = 0; // sliding window attention (SWA)
    uint32_t n_swa_pattern = 1; // by default, all layers use non-sliding-window attention
    uint32_t n_embd_head_k; // dimension of keys (d_k). d_q is assumed to be the same, but there are n_head q heads, and only n_head_kv k-v heads
    uint32_t n_embd_head_v; // dimension of values (d_v) aka n_embd_head
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_rel_attn_bkts = 0;

    // for WavTokenizer
    struct llama_hparams_posnet   posnet;
    struct llama_hparams_convnext convnext;

    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_kv_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_ff_arr;

    uint32_t n_layer_dense_lead = 0;
    uint32_t n_lora_q           = 0;
    uint32_t n_lora_kv          = 0;
    uint32_t n_ff_exp           = 0;
    uint32_t n_ff_shexp         = 0;
    uint32_t n_expert_shared    = 0;
    uint32_t n_norm_groups      = 0;

    float    expert_weights_scale = 0.0;
    bool     expert_weights_norm  = false;
    uint32_t expert_gating_func   = LLAMA_EXPERT_GATING_FUNC_TYPE_NONE;

    float f_norm_eps;
    float f_norm_rms_eps;
    float f_norm_group_eps;

    float f_attn_logit_softcapping  = 50.0f;
    float f_final_logit_softcapping = 30.0f;

    // for RWKV
    uint32_t rescale_every_n_layers = 0;
    uint32_t time_mix_extra_dim     = 0;
    uint32_t time_decay_extra_dim   = 0;
    uint32_t wkv_head_size          = 0;
    uint32_t token_shift_count      = 2;
    uint32_t n_lora_decay           = 0;
    uint32_t n_lora_iclr            = 0;
    uint32_t n_lora_value_res_mix   = 0;
    uint32_t n_lora_gate            = 0;

    float    rope_attn_factor = 1.0f;
    float    rope_freq_base_train;
    float    rope_freq_base_train_swa;
    float    rope_freq_scale_train;
    float    rope_freq_scale_train_swa;
    uint32_t n_ctx_orig_yarn;
    float    rope_yarn_log_mul;

    std::array<int, 4> rope_sections;

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

    uint32_t n_head(uint32_t il = 0) const;

    uint32_t n_head_kv(uint32_t il = 0) const;

    uint32_t n_ff(uint32_t il = 0) const;

    uint32_t n_gqa(uint32_t il = 0) const;

    // dimension of key embeddings across all k-v heads
    uint32_t n_embd_k_gqa(uint32_t il = 0) const;

    // dimension of value embeddings across all k-v heads
    uint32_t n_embd_v_gqa(uint32_t il = 0) const;

    // dimension of the rolling state embeddings
    // corresponds to Mamba's conv_states size or RWKV's token_shift states size
    uint32_t n_embd_k_s() const;

    // dimension of the recurrent state embeddings
    uint32_t n_embd_v_s() const;

    bool is_swa(uint32_t il) const;
};

struct llama_model {
    llm_type type = LLM_TYPE_UNKNOWN;
    llm_arch arch = LLM_ARCH_UNKNOWN;

    std::string name = "n/a";

    llama_hparams hparams = {};
    llama_vocab   vocab;

    struct ggml_tensor * tok_embd   = nullptr;
    struct ggml_tensor * type_embd  = nullptr;
    struct ggml_tensor * pos_embd   = nullptr;
    struct ggml_tensor * tok_norm   = nullptr;
    struct ggml_tensor * tok_norm_b = nullptr;

    struct ggml_tensor * output_norm     = nullptr;
    struct ggml_tensor * output_norm_b   = nullptr;
    struct ggml_tensor * output          = nullptr;
    struct ggml_tensor * output_b        = nullptr;
    struct ggml_tensor * output_norm_enc = nullptr;

    // classifier
    struct ggml_tensor * cls       = nullptr;
    struct ggml_tensor * cls_b     = nullptr;
    struct ggml_tensor * cls_out   = nullptr;
    struct ggml_tensor * cls_out_b = nullptr;

    struct ggml_tensor * conv1d = nullptr;
    struct ggml_tensor * conv1d_b = nullptr;
};

struct llama_vocab_hack {
    struct token_data {
        std::string      text;
        float            score;
        llama_token_attr attr;
    };

    llama_vocab_hack();
    ~llama_vocab_hack();

    void load(llama_model_loader & ml, const LLM_KV & kv);

    enum llama_vocab_type     get_type()     const;
    enum llama_vocab_pre_type get_pre_type() const;

    uint32_t n_tokens() const;
    uint32_t n_token_types() const;

    std::string type_name() const;

    bool is_normal      (llama_token id) const;
    bool is_unknown     (llama_token id) const;
    bool is_control     (llama_token id) const;
    bool is_byte        (llama_token id) const;
    bool is_user_defined(llama_token id) const;
    bool is_unused      (llama_token id) const;
    bool is_eog         (llama_token id) const;

    uint8_t     token_to_byte(llama_token id) const;
    llama_token byte_to_token(uint8_t ch)     const;

    llama_token text_to_token(const std::string & text) const;

    const token_data & get_token_data(llama_token id) const;

    const char *     token_get_text (llama_token id) const;
    float            token_get_score(llama_token id) const;
    llama_token_attr token_get_attr (llama_token id) const;

    llama_token token_bos() const;
    llama_token token_eos() const;
    llama_token token_eot() const;
    llama_token token_eom() const;
    llama_token token_unk() const;
    llama_token token_sep() const;
    llama_token token_nl () const;
    llama_token token_pad() const;

    llama_token token_prefix() const;
    llama_token token_middle() const;
    llama_token token_suffix() const;

    llama_token token_fim_pre() const;
    llama_token token_fim_suf() const;
    llama_token token_fim_mid() const;
    llama_token token_fim_pad() const;
    llama_token token_fim_rep() const;
    llama_token token_fim_sep() const;

    bool get_add_space_prefix          () const;
    bool get_add_bos                   () const;
    bool get_add_eos                   () const;
    bool get_ignore_merges             () const;
    bool get_clean_spaces              () const;
    bool get_remove_extra_whitespaces  () const;
    bool get_escape_whitespaces        () const;
    bool get_treat_whitespace_as_suffix() const;

    int max_token_len() const;

    int find_bpe_rank(const std::string & token_left, const std::string & token_right) const;

    int32_t tokenize(
                   const char * text,
                      int32_t   text_len,
                  llama_token * tokens,
                      int32_t   n_tokens_max,
                         bool   add_special,
                         bool   parse_special) const;

    std::vector<llama_token> tokenize(
            const std::string & raw_text,
                         bool   add_special,
                         bool   parse_special = false) const;

    // does not write null-terminator to buf
    int32_t token_to_piece(
                  llama_token   token,
                         char * buf,
                      int32_t   length,
                      int32_t   lstrip,
                         bool   special) const;

    // use cached data
    const std::string & token_to_piece(llama_token token) const;

    int32_t detokenize(
            const llama_token * tokens,
                      int32_t   n_tokens,
                         char * text,
                      int32_t   text_len_max,
                         bool   remove_special,
                         bool   unparse_special) const;

    std::string detokenize(
            const std::vector<llama_token> & tokens,
                                      bool   special) const;

    void print_info() const;

    struct impl {
	    uint32_t n_token_types = 0; // for BERT-style token types

	    enum llama_vocab_type     type     = LLAMA_VOCAB_TYPE_SPM;
	    enum llama_vocab_pre_type pre_type = LLAMA_VOCAB_PRE_TYPE_DEFAULT;

	    int max_token_len = 0; // used for optimizing longest token search

	    // default LLaMA special tokens
	    // TODO: should we set all of these to LLAMA_TOKEN_NULL?
	    llama_token special_bos_id  = 1;
	    llama_token special_eos_id  = 2;
	    llama_token special_eot_id  = LLAMA_TOKEN_NULL;
	    llama_token special_eom_id  = LLAMA_TOKEN_NULL;
	    llama_token special_unk_id  = 0;
	    llama_token special_sep_id  = LLAMA_TOKEN_NULL;
	    llama_token special_pad_id  = LLAMA_TOKEN_NULL;
	    llama_token special_mask_id = LLAMA_TOKEN_NULL;

	    llama_token linefeed_id = 13;

	    // fim tokens
	    llama_token special_fim_pre_id = LLAMA_TOKEN_NULL;
	    llama_token special_fim_suf_id = LLAMA_TOKEN_NULL;
	    llama_token special_fim_mid_id = LLAMA_TOKEN_NULL;
	    llama_token special_fim_pad_id = LLAMA_TOKEN_NULL;
	    llama_token special_fim_rep_id = LLAMA_TOKEN_NULL; // repo
	    llama_token special_fim_sep_id = LLAMA_TOKEN_NULL; // file separator

	    // tokenizer flags
	    bool add_space_prefix           = false;
	    bool add_bos                    = false;
	    bool add_eos                    = false;
	    bool ignore_merges              = false;
	    bool clean_spaces               = false;  // clean_up_tokenization_spaces
	    bool remove_extra_whitespaces   = false;
	    bool escape_whitespaces         = true;
	    bool treat_whitespace_as_suffix = false;

	    std::unordered_map<std::string, llama_token> token_to_id;
	    std::vector<token_data>                      id_to_token;
	};
    std::unique_ptr<impl> pimpl;
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
    llama_vocab_hack* vocab = (llama_vocab_hack*) const_cast<llama_vocab*>(llama_model_get_vocab( model ));
    vocab->pimpl->id_to_token.resize( tensor->ne[1] );
    // *const_cast<uint32_t*>(&model->hparams.n_vocab) = tensor->ne[1];
}
/* END VALL-E SPECIFIC HELPERS */

/* End cringe code */