#pragma once

// C++ deps
#include <string>
#include <vector>
#include <unordered_map>

// external deps
#include <llama.h>
#include <encodec.h>
#include <dr_wav.h>
#include <espeak-ng/speak_lib.h>

// to-do: copy over import/export stuff from engine project (because I don't remember how I set it up in <uf/config.h>)
#define VALL_E_API

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

const int MODALITY_AR_NAR = 0;
const int MODALITY_NAR_LEN = 1;

const int MAX_DURATION = 75 * 12;
const int CTX_SIZE = 2048;
const int N_THREADS = 8;
const int N_GPU_LAYERS = 99;

typedef llama_token token_t;
typedef std::vector<std::vector<token_t>> vall_e_audio_codes_t;

// stores embeddings + metadata for an embedding range
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

struct score_t {
	int32_t idx;
	float value;

	bool operator<( const score_t& that ) const { return this->value < that.value; }
};

struct merge_entry_t {
	std::u32string pre;
	std::u32string post;
	std::u32string resolved;

	token_t pre_token;
	token_t post_token;
	token_t resolved_token;
};

struct vall_e_context_params_t {
	std::string model_path = "./data/vall_e.gguf";
	std::string encodec_path = "./data/encodec.bin";
	int32_t gpu_layers = N_GPU_LAYERS;
	int32_t n_threads = N_THREADS;
	int32_t ctx_size = CTX_SIZE;
	bool verbose = false;
};
struct vall_e_args_t {
	std::string text = "Hello world.";
	std::string prompt_path = "./data/prom.wav";
	std::string output_path = "./data/resp.wav";
	std::string language = "en";
	int modality = MODALITY_NAR_LEN;
	int max_steps = 30;
	int max_duration = 75 * 12;
};
// stores everything needed for vall_e.cpp
struct vall_e_context_t {
	vall_e_context_params_t params;

	io_map_t io_map;

	struct {
		llama_model* model = NULL;
		llama_context* ctx = NULL;
	} llama;

	struct {
		encodec_context* ctx;
	} encodec;
};
// stores the raw inputs to be fed
struct vall_e_inputs_t {
	std::string task = "tts";

	std::vector<token_t> phn = {};
	token_t lang = 0;
	token_t rvq_l = 0;
	vall_e_audio_codes_t prom = {};
	vall_e_audio_codes_t resp = {};
};

// helper tensor functions
std::vector<float> VALL_E_API read_2d_tensor( struct ggml_tensor* tensor );
//ggml_tensor* VALL_E_API view_2d_tensor( ggml_tensor* tensor, int32_t start, int32_t end, int32_t dim = 0 ); // cringe method to keep in my pocket
ggml_tensor* VALL_E_API view_2d_tensor( ggml_context* ctx, ggml_tensor* tensor, int32_t start, int32_t end, int32_t dim = 0 );
void VALL_E_API print_tokens( const std::vector<token_t>& tokens, const std::string& prefix = "Tokens: " );

std::vector<std::vector<float>> VALL_E_API map_embeddings( const std::vector<token_t>& tokens, int n_embd, const float* embds );
std::vector<std::vector<float>> VALL_E_API sum_embeddings( const vall_e_audio_codes_t& input, int n_embd, int rvq_l, const float** embds, int mode = EMBEDDING_MODE_PROM );
std::vector<float> VALL_E_API soft_max( int n_logits, const float* logits );

// batch and inferencing
void VALL_E_API batch_add( llama_batch& batch, token_t id, int n_embd, const float* embds, llama_pos pos, bool output, const std::vector<llama_seq_id> & seq_ids = {0} );
void VALL_E_API fill_batch( llama_batch& batch, vall_e_inputs_t& input, io_map_t& inputs_map, int mode );
std::vector<token_t> VALL_E_API generate( vall_e_context_t* ctx, vall_e_inputs_t& input, int max_tokens, int mode, bool verbose = true );

//
std::vector<token_t> VALL_E_API phonemize( vall_e_context_t* ctx, const std::string& text, const std::string& language = "auto" );

// encodec helpers
std::vector<float> VALL_E_API read_audio_from_disk( const std::string& path );
void VALL_E_API write_audio_to_disk( const std::vector<float>& waveform, const std::string& path );

std::vector<std::vector<int32_t>> VALL_E_API encode_audio( struct encodec_context* ectx, const std::vector<float>& waveform );
std::vector<float> VALL_E_API decode_audio( struct encodec_context* ectx, const std::vector<std::vector<int32_t>>& codes_2d );

// model-accessing helpers
const io_t& VALL_E_API vall_e_inputs_map_get_embeddings( io_map_t& inputs_map, const std::string& name );
const float* VALL_E_API vall_e_inputs_map_get_embeddings_p( io_map_t& inputs_map, const std::string& name );
int32_t VALL_E_API vall_e_inputs_map_get_classifier_idx( io_map_t& inputs_map, const std::string& name );
void VALL_E_API vall_e_inputs_map_init( io_map_t&, llama_model* model );

// context management
void VALL_E_API vall_e_print_usage( char** argv, const vall_e_context_params_t& params, const vall_e_args_t& args );
bool VALL_E_API vall_e_args_parse( int argc, char** argv, vall_e_context_params_t& params, vall_e_args_t& args );
vall_e_context_t* VALL_E_API vall_e_load( const vall_e_context_params_t& params );
vall_e_inputs_t vall_e_prepare_inputs( vall_e_context_t* ctx, const std::string& text, const std::string& prompt_path, const std::string& lang );
vall_e_audio_codes_t vall_e_generate( vall_e_context_t* ctx, vall_e_inputs_t& inputs, int modality = MODALITY_NAR_LEN );
void VALL_E_API vall_e_free( vall_e_context_t* ctx );