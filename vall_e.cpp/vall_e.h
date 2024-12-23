#pragma once

#include "llama.h"
#include "encodec.h"

#include "dr_wav.h"

#include <string>
#include <vector>
#include <unordered_map>

// to-do: copy over import/export stuff from engine project (because I don't remember how I set it up in <uf/config.h>)
#define VALL_E_API

#define LLAMA_CPP_EXTENDED 1 // whether the underlying llama.cpp has some extra functions
#define LLAMA_CPP_USE_VALL_E_ARCH 1 // whether the underlying llama.cpp is to use the VALL_E arch (or using LLAMA arch)

#if !LLAMA_CPP_EXTENDED
	#include "_llama.h" // cringe hotfix but I have to do this until llama.cpp's API exposes the tok_embd
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

// stores the raw inputs to be fed
struct input_t {
	std::string task = "tts";

	std::string phonemes = "";
	std::vector<llama_token> phn = {};
	llama_token lang = 0;
	llama_token rvq_l = 0;
	std::vector<std::vector<llama_token>> prom = {};
	std::vector<std::vector<llama_token>> resp = {};
};

// reference mapping from vall_e.export.py
/*
	[(0, 256), 'text_emb.weight', 'classifiers.proj.9.weight', None],
	[(256, 264), 'rvq_l_emb.weight', None, '<|RVQ:{l}|>'],
	[(264, 270), 'langs_emb.weight', None, '<|lang:{lang}|>'],
	[(270, 279), 'tasks_emb.weight', None, '<|task:{task}|>'],
	[(279, 290), 'len_emb.weight', 'classifiers.proj.10.weight', '<|len:{id}|>'],
	[(290, 291), 'tones_emb.weight', None, '<|tone:{tone}|>'],
	[(291, 292), 'sep', None, '<|sep|>'],
	[(292, 1316), 'proms_emb.embeddings.0.weight', None, '<|P|0|{id}|>'],
	[(1316, 2340), 'proms_emb.embeddings.1.weight', None, '<|P|1|{id}|>'],
	[(2340, 3364), 'proms_emb.embeddings.2.weight', None, '<|P|2|{id}|>'],
	[(3364, 4388), 'proms_emb.embeddings.3.weight', None, '<|P|3|{id}|>'],
	[(4388, 5412), 'proms_emb.embeddings.4.weight', None, '<|P|4|{id}|>'],
	[(5412, 6436), 'proms_emb.embeddings.5.weight', None, '<|P|5|{id}|>'],
	[(6436, 7460), 'proms_emb.embeddings.6.weight', None, '<|P|6|{id}|>'],
	[(7460, 8484), 'proms_emb.embeddings.7.weight', None, '<|P|7|{id}|>'],
	[(8484, 9509), 'resps_emb.embeddings.0.weight', 'classifiers.proj.0.weight', '<|R|AR|0:0|{id}|>'],
	[(9509, 10533), 'resps_emb.embeddings.1.weight', 'classifiers.proj.1.weight', '<|R|NAR|0:1|{id}|>'],
	[(10533, 11557), 'resps_emb.embeddings.2.weight', 'classifiers.proj.2.weight', '<|R|NAR|1:2|{id}|>'],
	[(11557, 12581), 'resps_emb.embeddings.3.weight', 'classifiers.proj.3.weight', '<|R|NAR|2:3|{id}|>'],
	[(12581, 13605), 'resps_emb.embeddings.4.weight', 'classifiers.proj.4.weight', '<|R|NAR|3:4|{id}|>'],
	[(13605, 14629), 'resps_emb.embeddings.5.weight', 'classifiers.proj.5.weight', '<|R|NAR|4:5|{id}|>'],
	[(14629, 15653), 'resps_emb.embeddings.6.weight', 'classifiers.proj.6.weight', '<|R|NAR|5:6|{id}|>'],
	[(15653, 16677), 'resps_emb.embeddings.7.weight', 'classifiers.proj.7.weight', '<|R|NAR|6:7|{id}|>'],
	[(16677, 17702), 'resps_emb.embeddings.8.weight', 'classifiers.proj.8.weight', '<|R|NAR|0:0|{id}|>']
*/

// handles all the cringe logic of slicing embeddings
struct ranges_t {
	std::string name;

	uint32_t start;
	uint32_t end;
	
	int32_t classifier_idx = -1;
};

// stores embeddings + metadata for an embedding range
struct embeddings_t {
	int32_t n_embd = 0;
	int32_t n_vocab = 0;

	ranges_t range = {};
	std::vector<float> embds = {};
};

// stores the mappings between tokens, input embeddings, and output heads
struct inputs_map_t {
	int32_t n_embd = 0;
	int32_t n_vocab = 0;
	
	// mapping
	std::unordered_map<std::string, embeddings_t> embds = {};
};

// helper tensor functions
std::vector<float> VALL_E_API read_2d_tensor( struct ggml_tensor* tensor );
std::vector<std::vector<float>> VALL_E_API map_embeddings( const std::vector<llama_token>& tokens, int n_embd, const float* embds );
std::vector<std::vector<float>> VALL_E_API sum_embeddings( const std::vector<std::vector<llama_token>>& input, int n_embd, int rvq_l, const float** embds, int mode = EMBEDDING_MODE_PROM );
std::vector<float> VALL_E_API soft_max( int n_logits, const float* logits );

// batch and inferencing
void VALL_E_API batch_add( llama_batch& batch, llama_token id, int n_embd, const float* embds, llama_pos pos, bool output, const std::vector<llama_seq_id> & seq_ids = {0} );
void VALL_E_API fill_batch( llama_batch& batch, input_t& input, inputs_map_t& inputs_map, int mode );
std::vector<llama_token> VALL_E_API generate( llama_context* ctx, llama_model* model, llama_sampler* smpl, input_t& input, inputs_map_t& inputs_map, int max_tokens, int mode, bool verbose = true );

// encodec helpers
bool VALL_E_API read_wav_from_disk( std::string in_path, std::vector<float>& audio_arr );
void VALL_E_API write_wav_on_disk( std::vector<float>& audio_arr, std::string dest_path );
std::vector<std::vector<int32_t>> VALL_E_API encode_audio_from_disk( struct encodec_context* ectx, const std::string& path );
std::vector<float> VALL_E_API decode_audio( struct encodec_context* ectx, const std::vector<std::vector<int32_t>>& codes_2d );

// model-accessing helpers
const embeddings_t& VALL_E_API vall_e_inputs_map_get_embeddings( inputs_map_t& inputs_map, const std::string& name );
const float* VALL_E_API vall_e_inputs_map_get_embeddings_p( inputs_map_t& inputs_map, const std::string& name );
int32_t VALL_E_API vall_e_inputs_map_get_classifier_idx( inputs_map_t& inputs_map, const std::string& name );
void VALL_E_API vall_e_inputs_map_init( inputs_map_t&, llama_model* model );

struct ggml_tensor * VALL_E_API vall_e_get_prom_embds( llama_vall_e_userdata& userdata, int32_t idx );
struct ggml_tensor * VALL_E_API vall_e_get_resp_embds( llama_vall_e_userdata& userdata, int32_t idx );
struct ggml_tensor * VALL_E_API vall_e_get_aux_embds( llama_vall_e_userdata& userdata, int32_t idx );