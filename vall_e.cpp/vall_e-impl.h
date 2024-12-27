#pragma once

// stores all the backend stuff

// external deps
#include <llama.h>
#include <encodec.h>
#include <dr_wav.h>
#include <espeak-ng/speak_lib.h>

#define LLAMA_CPP_EXTENDED 0 // whether the underlying llama.cpp has some extra functions
#define LLAMA_CPP_USE_VALL_E_ARCH 0 // whether the underlying llama.cpp is to use the VALL_E arch (or using LLAMA arch)

#if !LLAMA_CPP_EXTENDED
	#include "llama_hack.h" // cringe hotfix but I have to do this until llama.cpp's API exposes the tok_embd
#endif

// to-do: clean up spaghetti enums
const int EMBEDDING_MODE_PROM = 0;
const int EMBEDDING_MODE_RESP_AR_NAR = 1;
const int EMBEDDING_MODE_RESP_NAR_LEN = 2;

const int INFERENCE_MODE_LEN = 0;
const int INFERENCE_MODE_AR = 1;
const int INFERENCE_MODE_NAR_DEMASK = 2;
const int INFERENCE_MODE_NAR = 3;

// stores metadata for inputs/outputs
struct io_t {
	std::string name;
	uint32_t start;
	uint32_t end;	
	int32_t head_idx = -1;

	int32_t n_embd = 0;
	int32_t n_vocab = 0;

	std::vector<float> embds = {};
	ggml_tensor* head = NULL;
};

// stores the mappings between tokens, input embeddings, and output heads
struct io_map_t {
	// model's original params
	int32_t n_embd = 0;
	int32_t n_vocab = 0;
	
	// mapping
	std::unordered_map<std::string, io_t> io = {};
	// context to store slices
	ggml_context* ctx = NULL;
};
// used for top-k (mainly for demasking)
struct score_t {
	int32_t idx;
	float value;

	bool operator<( const score_t& that ) const { return this->value < that.value; }
};
// handles storing metadata for token merges
struct merge_entry_t {
	std::u32string pre;
	std::u32string post;
	std::u32string resolved;

	token_t pre_token;
	token_t post_token;
	token_t resolved_token;
};

// helper tensor functions
std::vector<float> read_2d_tensor( struct ggml_tensor* tensor );
//ggml_tensor* view_2d_tensor( ggml_tensor* tensor, int32_t start, int32_t end, int32_t dim = 0 ); // cringe method to keep in my pocket
ggml_tensor* view_2d_tensor( ggml_context* ctx, ggml_tensor* tensor, int32_t start, int32_t end, int32_t dim = 0 );
void print_tokens( const std::vector<token_t>& tokens, const std::string& prefix = "Tokens: " );

std::vector<std::vector<float>> map_embeddings( const std::vector<token_t>& tokens, int n_embd, const float* embds );
std::vector<std::vector<float>> sum_embeddings( const vall_e_audio_codes_t& input, int n_embd, int rvq_l, const float** embds, int mode = EMBEDDING_MODE_PROM );
std::vector<float> soft_max( int n_logits, const float* logits );

// batch and inferencing
void batch_add( llama_batch& batch, token_t id, int n_embd, const float* embds, llama_pos pos, bool output, const std::vector<llama_seq_id> & seq_ids = {0} );
void fill_batch( llama_batch& batch, vall_e_inputs_t& input, io_map_t& inputs_map, int mode );
std::vector<token_t> generate( vall_e_context_t* ctx, vall_e_inputs_t& input, int max_tokens, int mode, bool verbose = true );

// (handles text)
std::vector<token_t> phonemize( vall_e_context_t* ctx, const std::string& text, const std::string& language = "auto" );

// model-accessing helpers
const io_t& vall_e_inputs_map_get_embeddings( io_map_t& inputs_map, const std::string& name );
const float* vall_e_inputs_map_get_embeddings_p( io_map_t& inputs_map, const std::string& name );
int32_t vall_e_inputs_map_get_classifier_idx( io_map_t& inputs_map, const std::string& name );
void vall_e_inputs_map_init( io_map_t&, llama_model* model );