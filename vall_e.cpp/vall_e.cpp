#include "llama-vocab.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <iostream>

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
};
/* End cringe code */

// handles adding either a token OR the embedding of that token into the batch
// this really, really helps avoid needing to abuse the tokenizer
// to-do: handle summing
void batch_add( struct llama_batch& batch, llama_token id, int n_embd, float* embds, llama_pos pos, bool logits = true, const std::vector<llama_seq_id> & seq_ids = {0} ) {
	GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

	if ( embds ) {
		for ( auto i = 0; i < n_embd; ++i ) {
			batch.embd[batch.n_tokens + i] = embds[id * n_embd + i];
		}
	} else {
		batch.token[batch.n_tokens] = id;
	}
	batch.pos[batch.n_tokens] = pos;
	batch.n_seq_id[batch.n_tokens] = seq_ids.size();
	for (size_t i = 0; i < seq_ids.size(); ++i) {
		batch.seq_id[batch.n_tokens][i] = seq_ids[i];
	}
	batch.logits[batch.n_tokens] = logits;

	batch.n_tokens++;
}

int main(int argc, char ** argv) {
	bool is_ar = true;
	// to-do: replace all of this with proper loading code
	std::vector<llama_token> phoneme_tokens = {1,85,4,128,26,4,186,4,89,33,25,4,48,4,134,25,52,86,4,34,97,27,11,2};
	llama_token lang_token = 0;
	llama_token rvq_level_token = 0;
	std::vector<std::vector<llama_token>> prompt_tokens = {
		{780,835,835,835,339,395,798,537,537,537,537,222,76,989,548,65,705,375,261,375,297,503,529,571,707,346,464,862,148,496,574,115,115,438,934,339,865,876,63,40,779,461,602,794,10,220,398,869,639,705,869,917,705,893,215,705,869,938,439,175,139,506,375,529,297,705,651,238,962,461,195,441,377,581,473,795,644,626,459,981,767,670,696,73,779,257,408,1017,1019,133,133,1017,835,604,699,626,67,92,707,92,179,179,772,869,441,799,917,238,745,904,904,904,106,133,1019,1017,1017,395,883,87,519,594,1002,682,996,540,186,1019,430,202,347,889,61,92,542,297,67,669,571,707,346,67,359,571,707,669,604,25,1008,810,35,621,67,600,333,123,284,568,817,243,778,464,638,610,359,538,464,975,321,700,377,484,179,284,284,621,538,464,745,171,171,159,744,159,287,461,69,15,529,67,92,669,464,515,605,24,822,865,293,62,172,638,359,562,138,839,846,775,556,688,1006,917,297,312,148,331,496,646,67,314,15,705,131,855,662,287,172,85,538,519,762,450,391,609,643,778,80,287,794,794,115,785,794,461,699,519,932,522,652,262,508,902,932,932,391,769,18,507,90,442,762,610,610,669,605,310,855,56,989,863,195,464,604,257,904,632,786,951,461,239,195,878,771,146,481,146,481,434,643,917,280,67,464,115,744,744,115,115,115,819,709,63,368,359,519,996,616,464,996,616,519,762,917,841,772,568,954,600,422,893,592,464,626,86,143,615,171,744,744,196,115,821,415,521,799,654,839,644,473,592,953,523,855,738,855,876,876,1017,63,329},
	};
	std::vector<std::vector<llama_token>> response_tokens = {
		{922,395,869,869,354,989,762,762,762,610,975,626,626,866,609,442,762,762,762,610,610,610,610,212,869,869,51,336,352,352,352,570,148,893,76,535,568,568,270,568,568,560,597,86,744,744,744,203,738,408,1019,700,707,92,707,464,744,171,171,159,196,192,697,261,261,568,638,605,904,904,779,832,570,519,223,459,459,459,459,90,90,570,700,53,372,621,610,869,473,869,917,654,473,917,893,654,644,384,558,911,864,521,1,19,665},
	};
	std::string model_path = "./vall_e/Vall_E-238M-Q8_0.gguf";

	// load dynamic backends
	ggml_backend_load_all();

	// initialize the model
	llama_model_params model_params = llama_model_default_params();
	model_params.n_gpu_layers = 0;

	llama_model* model = llama_load_model_from_file(model_path.c_str(), model_params);
	if (model == NULL) {
		fprintf(stderr , "%s: error: unable to load model\n" , __func__);
		return 1;
	}

	// initialize the context
	llama_context_params ctx_params = llama_context_default_params();
	ctx_params.n_ctx = 22500;
	ctx_params.n_batch = 22500;
	ctx_params.no_perf = false;

	ctx_params.attention_type = is_ar ? LLAMA_ATTENTION_TYPE_CAUSAL : LLAMA_ATTENTION_TYPE_NON_CAUSAL;

	llama_context* ctx = llama_new_context_with_model(model, ctx_params);
	if (ctx == NULL) {
		fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
		return 1;
	}

	// initialize the sampler
	auto sparams = llama_sampler_chain_default_params();
	sparams.no_perf = false;
	llama_sampler * smpl = llama_sampler_chain_init(sparams);

	llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

	// prepare batch
	auto n_embd = llama_n_embd( model );
	auto n_vocab = llama_n_vocab( model );
	llama_batch batch = llama_batch_init( ctx_params.n_ctx, n_embd, ctx_params.n_ctx );

	// grab input embeddings	
	std::vector<float> embds( n_embd * n_vocab );
	auto* qtype = ggml_get_type_traits(model->tok_embd->type);
	// dequantize if needed
	if ( ggml_is_quantized(model->tok_embd->type) ) {
		qtype->to_float(model->tok_embd->data, embds.data(), embds.size());
	}

	// to-do: derive these offsets from the tokenizer itself
	// to-do: clean this up, probably make it at parity to inputs_to_embeddings
	int text_embd_start = 0; // <unk>
	int rvq_level_embd_start = 17666; // <|RVQ:0>
	int len_embd_start = 17674; // <|len:0|>
	int lang_embd_start = 17686; // <|lang:en|>
	int task_embd_start = 17692; // <|task:tts|>
	int sep_embd_start = 17685; // <|sep|>
	int prom_embd_start[] = {
		256 + (1024 * 0), // <|P|0:0|>
		256 + (1024 * 1), // <|P|1:0|>
		256 + (1024 * 2), // <|P|2:0|>
		256 + (1024 * 3), // <|P|3:0|>
		256 + (1024 * 4), // <|P|4:0|>
		256 + (1024 * 5), // <|P|5:0|>
		256 + (1024 * 6), // <|P|6:0|>
		256 + (1024 * 7), // <|P|7:0|>
	};
	int resp_embd_start[] = {
		8448, // <|AR|0:0|>
		9473, // <|NAR|0:0|>
		10498 + (1024 * 0), // <|NAR|0:1|>
		10498 + (1024 * 1), // <|NAR|1:2|>
		10498 + (1024 * 2), // <|NAR|2:3|>
		10498 + (1024 * 3), // <|NAR|3:4|>
		10498 + (1024 * 4), // <|NAR|4:5|>
		10498 + (1024 * 5), // <|NAR|5:6|>
		10498 + (1024 * 6), // <|NAR|6:7|>
	};

	float* text_embds = &embds[text_embd_start * n_embd];
	float* rvq_level_embd = &embds[rvq_level_embd_start * n_embd];
	float* len_embd = &embds[len_embd_start * n_embd];
	float* lang_embd = &embds[lang_embd_start * n_embd];
	float* task_embd = &embds[task_embd_start * n_embd];
	float* sep_embd = &embds[sep_embd_start * n_embd];

	float* prom_embds[] = {
		&embds[prom_embd_start[0] * n_embd],
		&embds[prom_embd_start[1] * n_embd],
		&embds[prom_embd_start[2] * n_embd],
		&embds[prom_embd_start[3] * n_embd],
		&embds[prom_embd_start[4] * n_embd],
		&embds[prom_embd_start[5] * n_embd],
		&embds[prom_embd_start[6] * n_embd],
		&embds[prom_embd_start[7] * n_embd],
	};
	float* resps_embds[] = {
		&embds[resp_embd_start[0] * n_embd],
		&embds[resp_embd_start[1] * n_embd],
		&embds[resp_embd_start[2] * n_embd],
		&embds[resp_embd_start[3] * n_embd],
		&embds[resp_embd_start[4] * n_embd],
		&embds[resp_embd_start[5] * n_embd],
		&embds[resp_embd_start[6] * n_embd],
		&embds[resp_embd_start[7] * n_embd],
		&embds[resp_embd_start[8] * n_embd],
	};

	// insert into batch
	{ 
		// keeps track of the position for each sequence
		size_t pos = 0;
		
		// insert text tokens
		for ( auto& id : phoneme_tokens ) batch_add( batch, id, n_embd, text_embds, pos++, false );
		batch_add( batch, 0, n_embd, sep_embd, pos++, false );
		pos = 0;
		// insert lang token
		batch_add( batch, lang_token, n_embd, lang_embd, pos++, false );
		batch_add( batch, 0, n_embd, sep_embd, pos++, false );
		pos = 0;
		// insert rvq level token
		batch_add( batch, rvq_level_token, n_embd, rvq_level_embd, pos++, false );
		batch_add( batch, 0, n_embd, sep_embd, pos++, false );
		pos = 0;
		// insert prom tokens
		// to-do: handle summing
		for ( auto l = 0; l < prompt_tokens.size(); ++l ) {
			for ( auto& id : prompt_tokens[l] ) batch_add( batch, id, n_embd, prom_embds[l], pos++, false );
		}
		batch_add( batch, 0, n_embd, sep_embd, pos++, is_ar );
		pos = 0;

		// fill in masked tokens
		if ( !is_ar ) {
			for ( auto i = 0; i < response_tokens[0].size(); ++i ) batch_add( batch, response_tokens[0][i], n_embd, resps_embds[1], pos++, true );
		}
		pos = 0;
	}

	// Decoding loop
	const auto t_main_start = ggml_time_us();
	int n_decode = 0;

	// to-do: handle other levels
	std::vector<llama_token> resps_tokens;
	while ( resps_tokens.size() < 32 ) {
		if (llama_decode(ctx, batch)) {
			fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
			return 1;
		}
		n_decode += 1;

		// align to AR's classifier
		// to-do: derive from tokenizer
		int range[] = { resp_embd_start[0], resp_embd_start[1] };
		auto* logits = llama_get_logits_ith( ctx, -1 );
		for ( auto i = 0; i < n_vocab; ++i ) {
			if ( i < range[0] || i >= range[1] ) {
				logits[i] = -INFINITY;
			}
		}

		// sample the next token
		auto t = llama_sampler_sample(smpl, ctx, -1);

		// is stop token
		if ( t == resp_embd_start[1] - 1 ) { // <|AR|0:STOP|>
			break;
		}

		char buf[256];
		llama_token_to_piece( model, t, buf, sizeof(buf), 0, true );
		printf("%s\n", buf );

		batch_add( batch, 0, n_embd, resps_embds[0], resps_tokens.size(), true );
		resps_tokens.emplace_back(t);
	}
	printf("\n");

	const auto t_main_end = ggml_time_us();

	fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
			__func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

	fprintf(stderr, "\n");
	llama_perf_sampler_print(smpl);
	llama_perf_context_print(ctx);
	fprintf(stderr, "\n");

	llama_sampler_free(smpl);
	llama_free(ctx);
	llama_free_model(model);

	return 0;
}