#include "llama-vocab.h"
#include "llama.h"
#include "encodec.h"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <iostream>

#define LLAMA_CPP_EXTENDED 1 // whether the underlying llama.cpp has some extra functions
#define LLAMA_CPP_USE_VALL_E_ARCH 0 // whether the underlying llama.cpp is to use the VALL_E arch

#if !LLAMA_CPP_EXTENDED
	#include "_llama.h" // cringe hotfix but I have to do this until llama.cpp's API exposes the tok_embd
#endif

std::vector<float> read_2d_tensor( struct ggml_tensor* tensor ) {
	size_t size = tensor->ne[0] * tensor->ne[1];
	std::vector<float> res( size );
	
	auto* qtype = ggml_get_type_traits(tensor->type);
	// dequantize if needed
	if ( ggml_is_quantized(tensor->type) ) {
		qtype->to_float(tensor->data, res.data(), res.size());
	} else {
		memcpy( res.data(), tensor->data, res.size() * sizeof(float) );
	}

	return res;
}

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
ranges_t io_ranges[] = {
	{ "text", 0, 256, 9, }, 
	{ "rvq_l", 256, 264, -1, }, 
	{ "lang", 264, 270, -1, }, 
	{ "task", 270, 279, -1, }, 
	{ "len", 279, 290, 10, }, 
	{ "tone", 290, 291, -1, }, 
	{ "sep", 291, 292, -1, }, 

	{ "prom|0", 292, 1316, -1, }, 
	{ "prom|1", 1316, 2340, -1, }, 
	{ "prom|2", 2340, 3364, -1, }, 
	{ "prom|3", 3364, 4388, -1, }, 
	{ "prom|4", 4388, 5412, -1, }, 
	{ "prom|5", 5412, 6436, -1, }, 
	{ "prom|6", 6436, 7460, -1, }, 
	{ "prom|7", 7460, 8484, -1, }, 

	{ "resps|AR:0 8484, 9509, 0,:0", }, 
	{ "resps|NAR:0 9509, 10533, 1,:1", }, 
	{ "resps|NAR:1: 10533, 11557, 2,2", }, 
	{ "resps|NAR:2: 11557, 12581, 3,3", }, 
	{ "resps|NAR:3: 12581, 13605, 4,4", }, 
	{ "resps|NAR:4: 13605, 14629, 5,5", }, 
	{ "resps|NAR:5: 14629, 15653, 6,6", }, 
	{ "resps|NAR:6: 15653, 16677, 7,7", }, 
	{ "resps|NAR:0: 16677, 17702, 8,0", }, 
};

struct embeddings_t {
	int n_embd;
	int n_vocab;

	ranges_t range;
	std::vector<float> embds;
};
struct embeddings_map_t {
	int n_embd = 0;
	int n_vocab = 0;
	
	// mapping
	std::unordered_map<std::string, embeddings_t> mapped_embeddings;

	const embeddings_t& get_embeddings( const std::string& name ) {
		return mapped_embeddings[name];
	}
	const float* get_embeddings_p( const std::string& name ) {
		return mapped_embeddings[name].embds.data();	
	}

	void init( llama_model* model ) {
		this->n_embd = llama_n_embd( model );
		this->n_vocab = llama_n_vocab( model );

	// to-do: figure a nicer way to do this
	#if LLAMA_CPP_USE_VALL_E_ARCH
		mapped_embeddings["text"] = { n_embd, 0, { "text", 0, 0, 9, }, read_2d_tensor(llama_get_vall_e_aux_embds(model, 0)) };
		mapped_embeddings["rvq_l"] = { n_embd, 0, { "rvq_l", 0, 0, -1, }, read_2d_tensor(llama_get_vall_e_aux_embds(model, 1)) };
		mapped_embeddings["langs"] = { n_embd, 0, { "langs", 0, 0, -1, }, read_2d_tensor(llama_get_vall_e_aux_embds(model, 2)) };
		mapped_embeddings["tasks"] = { n_embd, 0, { "tasks", 0, 0, -1, }, read_2d_tensor(llama_get_vall_e_aux_embds(model, 3)) };
		mapped_embeddings["len"] = { n_embd, 0, { "len", 0, 0, 10, }, read_2d_tensor(llama_get_vall_e_aux_embds(model, 4)) };
		mapped_embeddings["tones"] = { n_embd, 0, { "tones", 0, 0, -1, }, read_2d_tensor(llama_get_vall_e_aux_embds(model, 5)) };
		mapped_embeddings["sep"] = { n_embd, 0, { "sep", 0, 0, -1, }, read_2d_tensor(llama_get_vall_e_aux_embds(model, 6)) };

		mapped_embeddings["prom|0"] = { n_embd, 0, { "prom|0", 0, 0, -1, }, read_2d_tensor(llama_get_vall_e_prom_embds(model, 0)) };
		mapped_embeddings["prom|1"] = { n_embd, 0, { "prom|1", 0, 0, -1, }, read_2d_tensor(llama_get_vall_e_prom_embds(model, 1)) };
		mapped_embeddings["prom|2"] = { n_embd, 0, { "prom|2", 0, 0, -1, }, read_2d_tensor(llama_get_vall_e_prom_embds(model, 2)) };
		mapped_embeddings["prom|3"] = { n_embd, 0, { "prom|3", 0, 0, -1, }, read_2d_tensor(llama_get_vall_e_prom_embds(model, 3)) };
		mapped_embeddings["prom|4"] = { n_embd, 0, { "prom|4", 0, 0, -1, }, read_2d_tensor(llama_get_vall_e_prom_embds(model, 4)) };
		mapped_embeddings["prom|5"] = { n_embd, 0, { "prom|5", 0, 0, -1, }, read_2d_tensor(llama_get_vall_e_prom_embds(model, 5)) };
		mapped_embeddings["prom|6"] = { n_embd, 0, { "prom|6", 0, 0, -1, }, read_2d_tensor(llama_get_vall_e_prom_embds(model, 6)) };
		mapped_embeddings["prom|7"] = { n_embd, 0, { "prom|7", 0, 0, -1, }, read_2d_tensor(llama_get_vall_e_prom_embds(model, 7)) };
			
		mapped_embeddings["resps|AR:0:0"] = { n_embd, 0, { "resps|AR:0:0", 0, 0, 0, }, read_2d_tensor(llama_get_vall_e_resp_embds(model, 0)) };
		mapped_embeddings["resps|NAR:0:1"] = { n_embd, 0, { "resps|NAR:0:1", 0, 0, 1, }, read_2d_tensor(llama_get_vall_e_resp_embds(model, 1)) };
		mapped_embeddings["resps|NAR:1:2"] = { n_embd, 0, { "resps|NAR:1:2", 0, 0, 2, }, read_2d_tensor(llama_get_vall_e_resp_embds(model, 2)) };
		mapped_embeddings["resps|NAR:2:3"] = { n_embd, 0, { "resps|NAR:2:3", 0, 0, 3, }, read_2d_tensor(llama_get_vall_e_resp_embds(model, 3)) };
		mapped_embeddings["resps|NAR:3:4"] = { n_embd, 0, { "resps|NAR:3:4", 0, 0, 4, }, read_2d_tensor(llama_get_vall_e_resp_embds(model, 4)) };
		mapped_embeddings["resps|NAR:4:5"] = { n_embd, 0, { "resps|NAR:4:5", 0, 0, 5, }, read_2d_tensor(llama_get_vall_e_resp_embds(model, 5)) };
		mapped_embeddings["resps|NAR:5:6"] = { n_embd, 0, { "resps|NAR:5:6", 0, 0, 6, }, read_2d_tensor(llama_get_vall_e_resp_embds(model, 6)) };
		mapped_embeddings["resps|NAR:6:7"] = { n_embd, 0, { "resps|NAR:6:7", 0, 0, 7, }, read_2d_tensor(llama_get_vall_e_resp_embds(model, 7)) };
		mapped_embeddings["resps|NAR:0:0"] = { n_embd, 0, { "resps|NAR:0:0", 0, 0, 8, }, read_2d_tensor(llama_get_vall_e_resp_embds(model, 8)) };

		// update values
		for ( auto& pair : mapped_embeddings ) {
			auto& k = pair.first;
			auto& v = pair.second;
			auto& embds = v.embds;

			v.n_vocab = embds.size() / n_embd;
			v.range.end = v.n_vocab;
		}
	#else

	#if LLAMA_CPP_EXTENDED
		auto* tensor = llama_get_embedding_weights( model );
	#else
		auto* tensor = model->tok_embd;
	#endif

		// prepare slices
		std::vector<float> raw_embeddings = read_2d_tensor( tensor );
		for ( auto& range : io_ranges ) {
			mapped_embeddings[range.name] = {
				n_embd,
				range.end - range.start,
				range,
				std::vector<float>( raw_embeddings.data() + range.start, raw_embeddings.data() + range.end )
			};
		}
	#endif
	}
};

// maps embeddings easily
std::vector<std::vector<float>> map_embeddings( const std::vector<llama_token>& tokens, int n_embd, const float* embds ) {
	std::vector<std::vector<float>> embedded( tokens.size() );
	for ( auto i = 0; i < tokens.size(); ++i ) {
		embedded[i].insert( embedded[i].end(), embds + (tokens[i] * n_embd), embds + ((tokens[i]+1) * n_embd) );
	}
	return embedded;
}

// handles adding either a token OR the embedding of that token into the batch
// this really, really helps avoid needing to abuse the tokenizer
void batch_add( llama_batch& batch, llama_token id, int n_embd, const float* embds, llama_pos pos, bool output, const std::vector<llama_seq_id> & seq_ids = {0} ) {
	GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

	// insert raw embedding instead
	if ( embds ) {
		// signals to not map the embedding from the array
		if ( id < 0 ) for ( auto i = 0; i < n_embd; ++i ) batch.embd[batch.n_tokens + i] = embds[i];
		else for ( auto i = 0; i < n_embd; ++i ) batch.embd[batch.n_tokens + i] = embds[id * n_embd + i];
	// insert token (never gets used here)
	} else {
		batch.token[batch.n_tokens] = id;
	}

	batch.pos[batch.n_tokens] = pos;
	
	batch.n_seq_id[batch.n_tokens] = seq_ids.size();
	for (size_t i = 0; i < seq_ids.size(); ++i) batch.seq_id[batch.n_tokens][i] = seq_ids[i];
	batch.logits[batch.n_tokens] = output ? 1 : 0;
	
	// printf("[%i] Adding: %i | %i | %p | %i\n", batch.n_tokens, id, batch.pos[batch.n_tokens], &batch.embd[batch.n_tokens], batch.logits[batch.n_tokens] );
	// printf("[%i] Adding: %i | %i | %p | %i\n", batch.n_tokens, id, pos, embds, output );

	batch.n_tokens++;
}
// reads a waveform from disk
bool read_wav_from_disk(std::string in_path, std::vector<float> & audio_arr) {
    uint32_t channels;
    uint32_t sample_rate;
    drwav_uint64 total_frame_count;

    float * raw_audio = drwav_open_file_and_read_pcm_frames_f32(
        in_path.c_str(), &channels, &sample_rate, &total_frame_count, NULL);

    if (raw_audio == NULL) {
        fprintf(stderr, "%s: could not read wav file\n", __func__);
        return false;
    }

    if (sample_rate != 24000) {
        fprintf(stderr, "%s: wav file is wrong sample rate\n", __func__);
        return false;
    }

    fprintf(stderr, "\n%s: Number of frames read = %lld.\n", __func__, total_frame_count);

    audio_arr.resize(total_frame_count);
    memcpy(audio_arr.data(), raw_audio, total_frame_count * sizeof(float));

    drwav_free(raw_audio, NULL);

    return true;
}
// writes a waveform to disk
void write_wav_on_disk(std::vector<float> & audio_arr, std::string dest_path) {
    drwav_data_format format;
    format.bitsPerSample = 32;
    format.sampleRate = 24000;
    format.container = drwav_container_riff;
    format.channels = 1;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;

    drwav wav;
    drwav_init_file_write(&wav, dest_path.c_str(), &format, NULL);
    drwav_uint64 frames = drwav_write_pcm_frames(&wav, audio_arr.size(), audio_arr.data());
    drwav_uninit(&wav);

    fprintf(stderr, "%s: Number of frames written = %lld.\n", __func__, frames);
}
// reads a waveform from disk then encodes it
std::vector<std::vector<int32_t>> encode_audio_from_disk( struct encodec_context* ectx, const std::string& path ) {
	// read audio from disk
    std::vector<float> wavform;

    if(!read_wav_from_disk(path, wavform)) {
        printf("%s: error during reading wav file\n", __func__);
        return {};
    }

    // compress audio
    if (!encodec_compress_audio(ectx, wavform.data(), wavform.size(), 1)) {
        printf("%s: error during compression \n", __func__);
        return {};
    }

    int32_t* codes_data = encodec_get_codes( ectx );
    int n_codes = encodec_get_codes_size( ectx );
    int n_codebooks = 8;
    int n_frames = n_codes / n_codebooks;
    
    std::vector<int32_t> flattened_codes(codes_data, codes_data + n_codes);
    std::vector<std::vector<int32_t>> codes_2ds(8);

    for ( auto l = 0; l < n_codebooks; ++l ) {
    	codes_2ds[l].resize( n_frames );
    	for ( auto i = 0; i < n_frames; ++i ) {
			codes_2ds[l][i] = flattened_codes[i + l * n_codebooks];
    	}
    }

    return codes_2ds;
}
// decodes a 2D codebook into a waveform
std::vector<float> decode_audio( struct encodec_context* ectx, const std::vector<std::vector<int32_t>>& codes_2d ) {
    int n_codebooks = codes_2d.size();
    int n_frames = codes_2d[0].size();
	
	std::vector<int32_t> codes( n_frames * n_codebooks );
	
	for ( auto l = 0; l < n_codebooks; ++l ) {
		for ( auto i = 0; i < n_frames; ++i ) {
			codes[i + l * n_codebooks] = codes_2d[l][i];
		}
	}

    // decompress audio
    if (!encodec_decompress_audio(ectx, codes.data(), codes.size(), 1)) {
        printf("%s: error during decompression\n", __func__);
        return {};
    }

    // write reconstructed audio on disk
    const float* audio_data = encodec_get_audio(ectx);
    const int audio_size = encodec_get_audio_size(ectx);
    return std::vector<float>(audio_data, audio_data + audio_size);
}

const int EMBEDDING_MODE_PROM = 0;
const int EMBEDDING_MODE_RESP_AR_NAR = 1;
const int EMBEDDING_MODE_RESP_NAR_LEN = 2;

const int INFERENCE_MODE_LEN = 0;
const int INFERENCE_MODE_AR = 1;
const int INFERENCE_MODE_NAR_DEMASK = 2;
const int INFERENCE_MODE_NAR = 3;

const int MODALITY_AR_NAR = 0;
const int MODALITY_NAR_LEN = 1;

const int MAX_DURATION = 75; // * 12;
const int CTX_SIZE = 2048;

// sums embeddings over a 2D "tensor"
std::vector<std::vector<float>> sum_embeddings( const std::vector<std::vector<llama_token>>& input, int n_embd, int rvq_l, const float** embds, int mode = EMBEDDING_MODE_PROM ) {
	std::vector<std::vector<float>> res( input.size() );
	res.resize( input[0].size() );
	for ( auto& e : res ) e.resize( n_embd );
	// iterate through rvq levels (only up to inclusive the target rvq level)
	for ( auto l = 0; l < input.size() && l <= rvq_l; ++l ) {
		int offset = 0;
		// handles the cringe logic I have
		if ( mode == EMBEDDING_MODE_RESP_AR_NAR ) {
			offset = input.size() == 1 ? 0 : 1;
		} else if ( mode == EMBEDDING_MODE_RESP_NAR_LEN ) {
			offset = input.size() == 1 ? 8 : 1;
		}
		// get tokens
		auto& tokens = input[l];
		// get output buffer
		auto& summed = res[l];
		// embed the current level's tokens
		auto embedded = map_embeddings( input[l], n_embd, embds[l + offset] );
		// iterate through embedded tokens
		for ( auto i = 0; i < tokens.size(); ++i ) {
			// sum with buffer
			for ( auto j = 0; j < n_embd; ++j ) summed[j] += embedded[i][j];
		}
	}
	return res;
}

void fill_batch( llama_batch& batch, input_t& input, embeddings_map_t& embeddings_map, int mode ) {
	// keeps track of the position for each sequence
	size_t pos = 0;
	auto n_embd = embeddings_map.n_embd;

	const float* text_embds = embeddings_map.get_embeddings_p("text");
	const float* rvq_l_embds = embeddings_map.get_embeddings_p("rvq_l");
	const float* lang_embds = embeddings_map.get_embeddings_p("lang");
	const float* task_embds = embeddings_map.get_embeddings_p("task");
	const float* len_embds = embeddings_map.get_embeddings_p("len");
	const float* tone_embds = embeddings_map.get_embeddings_p("tone");
	const float* sep_embds = embeddings_map.get_embeddings_p("sep");
	const float* prom_embds[] = {
		embeddings_map.get_embeddings_p("prom|0"),
		embeddings_map.get_embeddings_p("prom|1"),
		embeddings_map.get_embeddings_p("prom|2"),
		embeddings_map.get_embeddings_p("prom|3"),
		embeddings_map.get_embeddings_p("prom|4"),
		embeddings_map.get_embeddings_p("prom|5"),
		embeddings_map.get_embeddings_p("prom|6"),
		embeddings_map.get_embeddings_p("prom|7"),
	};
	const float* resp_embds[] = {
		embeddings_map.get_embeddings_p("resps|AR:0:0"),
		embeddings_map.get_embeddings_p("resps|NAR:0:1"),
		embeddings_map.get_embeddings_p("resps|NAR:1:2"),
		embeddings_map.get_embeddings_p("resps|NAR:2:3"),
		embeddings_map.get_embeddings_p("resps|NAR:3:4"),
		embeddings_map.get_embeddings_p("resps|NAR:4:5"),
		embeddings_map.get_embeddings_p("resps|NAR:5:6"),
		embeddings_map.get_embeddings_p("resps|NAR:6:7"),
		embeddings_map.get_embeddings_p("resps|NAR:0:0"),
	};

	// insert text tokens
	for ( auto& id : input.phn ) batch_add( batch, id, n_embd, text_embds, pos++, false );
	batch_add( batch, 0, n_embd, sep_embds, pos++, false );
	pos = 0;
	// insert lang token
	batch_add( batch, input.lang, n_embd, lang_embds, pos++, false );
	batch_add( batch, 0, n_embd, sep_embds, pos++, false );
	pos = 0;
	// insert rvq level token
	batch_add( batch, input.rvq_l, n_embd, rvq_l_embds, pos++, false );
	batch_add( batch, 0, n_embd, sep_embds, pos++, false );
	pos = 0;
	// insert prom tokens
	auto summed_proms_embds = sum_embeddings( input.prom, n_embd, input.rvq_l, prom_embds );
	for ( auto i = 0; i < summed_proms_embds.size(); ++i ) {
		batch_add( batch, -1, n_embd, &summed_proms_embds[i][0], pos++, false );
	}
	batch_add( batch, 0, n_embd, sep_embds, pos++, mode == INFERENCE_MODE_AR ); // set as the last logit if AR
	pos = 0;

	// input starting len token
	if ( input.task == "len" ) {
		batch_add( batch, 0, n_embd, len_embds, pos++, true );
		pos = 0;
	}

	// insert resp tokens
	if ( !input.resp.empty() ) {
		auto summed_resps_embds = sum_embeddings( input.resp, n_embd, input.rvq_l, resp_embds, mode == INFERENCE_MODE_AR ? EMBEDDING_MODE_RESP_AR_NAR : EMBEDDING_MODE_RESP_NAR_LEN );
		for ( auto i = 0; i < summed_resps_embds.size(); ++i ) {
			batch_add( batch, -1, n_embd, &summed_resps_embds[i][0], pos++, true );
		}
		pos = 0;
	}
}

// generation code, should handle all modalities easily
std::vector<llama_token> generate( llama_context* ctx, llama_model* model, llama_sampler* smpl, input_t& input, embeddings_map_t& embeddings_map, int max_tokens, int mode, bool verbose = true ) {
	llama_batch batch = llama_batch_init( 22500, embeddings_map.n_embd, 22500 );

	// Decoding loop
	const auto t_main_start = ggml_time_us();
	int n_decode = 0;
	int rvq_l = input.rvq_l;
	llama_token stop_token = -1;
	
	fill_batch( batch, input, embeddings_map, mode );

	// determine how many logits we need
	int n_logits = 0;
	for ( auto i = 0; i < batch.n_tokens; ++i ) {
		if ( batch.logits[i] ) ++n_logits;
	}
	
	if ( verbose ) printf("Prompt size: %i | Outputs: %i\n", batch.n_tokens, n_logits);

	// NAR mode, cap at one step
	if ( n_logits > 1 ) {
		max_tokens = n_logits;
	}

	if ( n_logits == 0 ) {
		fprintf(stderr, "%s : no tokens to decode\n", __func__);
		return {};
	}

	const float* embds = NULL;
	ranges_t range;

	if ( mode == INFERENCE_MODE_AR ) {
		auto& embeddings = embeddings_map.get_embeddings("resps|AR:0:0");
		range = embeddings.range;
		embds = embeddings.embds.data();
		stop_token = range.end - 1;

		// llama_set_classifier_index( ctx, 0 );

		printf("Generating in %s mode (%i:%i)\n", "AR", range.start, range.end);
	} else if ( mode == INFERENCE_MODE_NAR ) {
		std::string k_embds[] = {
			"resps|NAR:0:0", // invalid
			"resps|NAR:0:1",
			"resps|NAR:1:2",
			"resps|NAR:2:3",
			"resps|NAR:3:4",
			"resps|NAR:4:5",
			"resps|NAR:5:6",
			"resps|NAR:6:7",
		};
		auto& embeddings = embeddings_map.get_embeddings(k_embds[rvq_l]);
		range = embeddings.range;
		embds = embeddings.embds.data();
		
		// llama_set_classifier_index( ctx, rvq_l );
		printf("Generating in %s mode (%i:%i)\n", "NAR", range.start, range.end);
	} else if ( mode == INFERENCE_MODE_LEN ) {
		auto& embeddings = embeddings_map.get_embeddings("len");
		range = embeddings.range;
		embds = embeddings.embds.data();
		stop_token = range.end - 1;
		
		// llama_set_classifier_index( ctx, 10 );
		printf("Generating in %s mode (%i:%i)\n", "len", range.start, range.end);
	} else if ( mode == INFERENCE_MODE_NAR_DEMASK ) {
		auto& embeddings = embeddings_map.get_embeddings("NAR:0:0");
		range = embeddings.range;
		embds = embeddings.embds.data();

		// llama_set_classifier_index( ctx, 8 );
		printf("Generating in %s mode (%i:%i)\n", "NAR-len", range.start, range.end);
	}

	llama_set_causal_attn( ctx, n_logits == 1 );

	std::vector<llama_token> output_tokens;
	while ( output_tokens.size() < max_tokens ) {
		if (llama_decode(ctx, batch)) {
			fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
			return output_tokens;
		}
		n_decode += 1;

		// backwards iterate to start from beginning of sequence
		for ( auto i = n_logits; i > 0; --i ) {
			// filter logits
			auto* logits = llama_get_logits_ith( ctx, -i );

		// ensures only tokens within our designated range are used			
		#if !LLAMA_CPP_USE_VALL_E_ARCH
			for ( auto i = 0; i < embeddings_map.n_vocab; ++i ) {
				if ( i < range.start || i >= range.end ) logits[i] = -INFINITY;
			}
		#endif

			// sample the next token
			auto t = llama_sampler_sample(smpl, ctx, -i);

		// offset back into range
		#if !LLAMA_CPP_USE_VALL_E_ARCH
			t -= range.start;
		#endif

			printf("%i: %i\n", n_decode, t);

			// is stop token
			if ( t == stop_token ) {
				printf("STOPPED\n");
				max_tokens = 0;
				break;
			}

			// store token
			output_tokens.emplace_back(t);
			// update batch with token
			batch_add( batch, t, embeddings_map.n_embd, embds, output_tokens.size(), true );
		}
	}
	const auto t_main_end = ggml_time_us();

	if ( verbose ) {
		printf("\n");
		fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
				__func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

		fprintf(stderr, "\n");
		llama_perf_sampler_print(smpl);
		llama_perf_context_print(ctx);
		fprintf(stderr, "\n");
	}
	
	llama_batch_free(batch);

	return output_tokens;
}

int main(int argc, char ** argv) {
	// to-do: replace all of this with proper loading code
	int32_t ngl = 0;
	int modality = MODALITY_NAR_LEN;
	input_t input{};
	embeddings_map_t embeddings_map{};

	// input.phonemes = "hˈɛloː ʋˈɔrlt";
	input.phn = {1,85,4,128,26,4,186,4,89,33,25,4,48,4,134,25,52,86,4,34,97,27,11,2}; // <bos>hˈɛloː ʋˈɔrlt</eos>

	std::string vall_e_model_path = "./data/vall_e-f16.gguf";
	std::string encodec_model_path = "./data/encodec.bin";
	std::string input_prompt_path = "./data/prom.wav";
	std::string output_response_path = "./data/resp.wav";

	// load dynamic backends
	ggml_backend_load_all();

	// initialize the models
	llama_model_params model_params = llama_model_default_params();
	model_params.n_gpu_layers = ngl;

	llama_model* model = llama_load_model_from_file(vall_e_model_path.c_str(), model_params);
	if (model == NULL) {
		fprintf(stderr , "%s: error: unable to load model\n" , __func__);
		return 1;
	}

	// initialize the context
	llama_context_params ctx_params = llama_context_default_params();
	ctx_params.n_ctx = CTX_SIZE;
	ctx_params.n_batch = CTX_SIZE;
	ctx_params.n_ubatch = CTX_SIZE;
	ctx_params.no_perf = false;
	ctx_params.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL; 

	// create two contexts, one's that causally, the other that isn't, because pain
	llama_context* ctx = llama_new_context_with_model(model, ctx_params);
	if (ctx == NULL) {
		fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
		return 1;
	}

	// initialize the sampler
	auto sparams = llama_sampler_chain_default_params();
	sparams.no_perf = false;
	llama_sampler * smpl_ar = llama_sampler_chain_init(sparams);
	llama_sampler * smpl_nar = llama_sampler_chain_init(sparams);

	llama_sampler_chain_add(smpl_ar, llama_sampler_init_top_k(20));
	llama_sampler_chain_add(smpl_ar, llama_sampler_init_top_p(0.9, 20));
	llama_sampler_chain_add(smpl_ar, llama_sampler_init_temp (1.0));
	llama_sampler_chain_add(smpl_ar, llama_sampler_init_dist (1130));
	
	llama_sampler_chain_add(smpl_nar, llama_sampler_init_greedy());

	struct encodec_context* ectx = encodec_load_model(encodec_model_path.c_str(), 0, ngl);
	if (!ectx) {
		printf("%s: error during loading model\n", __func__);
		return 1;
	}
	
	encodec_set_target_bandwidth(ectx, 6);
	encodec_set_sample_rate(ectx, 24000);

	// load wavform
	input.prom = encode_audio_from_disk(ectx, input_prompt_path);
	//input.resp = encode_audio_from_disk(ectx, output_response_path);

	// prepare batch
	auto n_embd = llama_n_embd( model );
	auto n_vocab = llama_n_vocab( model );

	// grab input embeddings	
	embeddings_map.init( model );

	// tokenize phonemes
	// to-do: make this work, the vocab does not work
	if ( input.phonemes != "" ) {
		const int n_prompt = -llama_tokenize(model, input.phonemes.c_str(), input.phonemes.size(), NULL, 0, true, true);
		// allocate space for the tokens and tokenize the input.phonemes
		input.phn.resize(n_prompt);
		if (llama_tokenize(model, input.phonemes.c_str(), input.phonemes.size(), input.phn.data(), input.phn.size(), true, true) < 0) {
		    fprintf(stderr, "%s: error: failed to tokenize: %s\n", __func__, input.phonemes.c_str());
		    return 1;
		}

		for ( auto& token : input.phn ) printf("%i ", token );
		printf("\n");
	}

	// inference
	std::vector<llama_token> output_tokens;
	// NAR-len demasking
	if ( modality == MODALITY_NAR_LEN ) {
		// inference len
		input.task = "len";
		output_tokens = generate( ctx, model, smpl_nar, input, embeddings_map, 5, INFERENCE_MODE_LEN );
		int len = 0; {
			int digit = 1;
			for (int i = output_tokens.size() - 1; i >= 0; i--) {
				len += output_tokens[i] * digit;
				digit *= 10;
			}
		}
		// cap for now
		if ( len <= 0 || len > MAX_DURATION ) len = MAX_DURATION;

		// fill with mask tokens
		input.resp.resize(1);
		for ( auto i = 0; i < len; ++i ) {
			input.resp[0].emplace_back( 1024 ); // fill with masked tokens
		}

		// inference NAR-len 0
		input.task = "tts";
		for ( auto l = 0; l < 8; ++l ) {
			input.rvq_l = l;
			output_tokens = generate( ctx, model, smpl_nar, input, embeddings_map, 5, l == 0 ? INFERENCE_MODE_NAR_DEMASK  : INFERENCE_MODE_NAR );
			input.resp.emplace_back( output_tokens );
		}
	// AR+NAR
	} else if ( modality == MODALITY_AR_NAR ){
		input.task = "tts";
		for ( auto l = 0; l < 8; ++l ) {
			input.rvq_l = l;
			output_tokens = generate( ctx, model, l == 0 ? smpl_ar : smpl_nar, input, embeddings_map, l == 0 ? MAX_DURATION : 1, l == 0 ? INFERENCE_MODE_AR  : INFERENCE_MODE_NAR );
			input.resp.emplace_back( output_tokens );
		}
	}

	// write audio to disk
	auto waveform = decode_audio( ectx, input.resp );
	write_wav_on_disk( waveform, output_response_path );

	// cleanup
	encodec_free(ectx);

	llama_sampler_free(smpl_nar);
	llama_sampler_free(smpl_ar);
	
	llama_free(ctx);
	
	llama_free_model(model);

	return 0;
}