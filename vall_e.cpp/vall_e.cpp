#define DR_WAV_IMPLEMENTATION
#include "vall_e.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

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

std::vector<float> VALL_E_API read_2d_tensor( struct ggml_tensor* tensor ) {
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


struct ggml_tensor * VALL_E_API  vall_e_get_prom_embds( llama_vall_e_userdata& userdata, int32_t idx ) {
    return userdata.prom_embds[idx];
}
struct ggml_tensor * VALL_E_API  vall_e_get_resp_embds( llama_vall_e_userdata& userdata, int32_t idx ) {
    return userdata.resp_embds[idx];
}
struct ggml_tensor * VALL_E_API  vall_e_get_aux_embds( llama_vall_e_userdata& userdata, int32_t idx ) {
    return userdata.aux_embds[idx];
}


const embeddings_t& VALL_E_API vall_e_inputs_map_get_embeddings( inputs_map_t& inputs_map, const std::string& name ) {
	return inputs_map.embds[name];
}
const float* VALL_E_API vall_e_inputs_map_get_embeddings_p( inputs_map_t& inputs_map, const std::string& name ) {
	return inputs_map.embds[name].embds.data();	
}

int32_t VALL_E_API vall_e_inputs_map_get_classifier_idx( inputs_map_t& inputs_map, const std::string& name ) {
	return inputs_map.embds[name].range.classifier_idx;
}

void VALL_E_API vall_e_inputs_map_init( inputs_map_t& inputs_map, llama_model* model ) {
	auto n_embd = llama_n_embd( model );
	auto n_vocab = llama_n_vocab( model );
	
	inputs_map.n_embd = n_embd;
	inputs_map.n_vocab = n_vocab;

	auto& userdata = *llama_get_vall_e_userdata( model );

// to-do: figure a nicer way to do this
#if LLAMA_CPP_USE_VALL_E_ARCH
	inputs_map.embds["text"] = { n_embd, 0, { "text", 0, 0, 9, }, read_2d_tensor(vall_e_get_aux_embds(userdata, 0)) };
	inputs_map.embds["rvq_l"] = { n_embd, 0, { "rvq_l", 0, 0, -1, }, read_2d_tensor(vall_e_get_aux_embds(userdata, 1)) };
	inputs_map.embds["lang"] = { n_embd, 0, { "lang", 0, 0, -1, }, read_2d_tensor(vall_e_get_aux_embds(userdata, 2)) };
	inputs_map.embds["task"] = { n_embd, 0, { "task", 0, 0, -1, }, read_2d_tensor(vall_e_get_aux_embds(userdata, 3)) };
	inputs_map.embds["len"] = { n_embd, 0, { "len", 0, 0, 10, }, read_2d_tensor(vall_e_get_aux_embds(userdata, 4)) };
	inputs_map.embds["tone"] = { n_embd, 0, { "tone", 0, 0, -1, }, read_2d_tensor(vall_e_get_aux_embds(userdata, 5)) };
	inputs_map.embds["sep"] = { n_embd, 0, { "sep", 0, 0, -1, }, read_2d_tensor(vall_e_get_aux_embds(userdata, 6)) };

	inputs_map.embds["prom|0"] = { n_embd, 0, { "prom|0", 0, 0, -1, }, read_2d_tensor(vall_e_get_prom_embds(userdata, 0)) };
	inputs_map.embds["prom|1"] = { n_embd, 0, { "prom|1", 0, 0, -1, }, read_2d_tensor(vall_e_get_prom_embds(userdata, 1)) };
	inputs_map.embds["prom|2"] = { n_embd, 0, { "prom|2", 0, 0, -1, }, read_2d_tensor(vall_e_get_prom_embds(userdata, 2)) };
	inputs_map.embds["prom|3"] = { n_embd, 0, { "prom|3", 0, 0, -1, }, read_2d_tensor(vall_e_get_prom_embds(userdata, 3)) };
	inputs_map.embds["prom|4"] = { n_embd, 0, { "prom|4", 0, 0, -1, }, read_2d_tensor(vall_e_get_prom_embds(userdata, 4)) };
	inputs_map.embds["prom|5"] = { n_embd, 0, { "prom|5", 0, 0, -1, }, read_2d_tensor(vall_e_get_prom_embds(userdata, 5)) };
	inputs_map.embds["prom|6"] = { n_embd, 0, { "prom|6", 0, 0, -1, }, read_2d_tensor(vall_e_get_prom_embds(userdata, 6)) };
	inputs_map.embds["prom|7"] = { n_embd, 0, { "prom|7", 0, 0, -1, }, read_2d_tensor(vall_e_get_prom_embds(userdata, 7)) };
		
	inputs_map.embds["resps|AR:0:0"] = { n_embd, 0, { "resps|AR:0:0", 0, 0, 0, }, read_2d_tensor(vall_e_get_resp_embds(userdata, 0)) };
	inputs_map.embds["resps|NAR:0:1"] = { n_embd, 0, { "resps|NAR:0:1", 0, 0, 1, }, read_2d_tensor(vall_e_get_resp_embds(userdata, 1)) };
	inputs_map.embds["resps|NAR:1:2"] = { n_embd, 0, { "resps|NAR:1:2", 0, 0, 2, }, read_2d_tensor(vall_e_get_resp_embds(userdata, 2)) };
	inputs_map.embds["resps|NAR:2:3"] = { n_embd, 0, { "resps|NAR:2:3", 0, 0, 3, }, read_2d_tensor(vall_e_get_resp_embds(userdata, 3)) };
	inputs_map.embds["resps|NAR:3:4"] = { n_embd, 0, { "resps|NAR:3:4", 0, 0, 4, }, read_2d_tensor(vall_e_get_resp_embds(userdata, 4)) };
	inputs_map.embds["resps|NAR:4:5"] = { n_embd, 0, { "resps|NAR:4:5", 0, 0, 5, }, read_2d_tensor(vall_e_get_resp_embds(userdata, 5)) };
	inputs_map.embds["resps|NAR:5:6"] = { n_embd, 0, { "resps|NAR:5:6", 0, 0, 6, }, read_2d_tensor(vall_e_get_resp_embds(userdata, 6)) };
	inputs_map.embds["resps|NAR:6:7"] = { n_embd, 0, { "resps|NAR:6:7", 0, 0, 7, }, read_2d_tensor(vall_e_get_resp_embds(userdata, 7)) };
	inputs_map.embds["resps|NAR:0:0"] = { n_embd, 0, { "resps|NAR:0:0", 0, 0, 8, }, read_2d_tensor(vall_e_get_resp_embds(userdata, 8)) };

	// update values
	for ( auto& pair : inputs_map.embds ) {
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
		inputs_map.embds[range.name] = {
			n_embd,
			range.end - range.start,
			range,
			std::vector<float>( raw_embeddings.data() + range.start, raw_embeddings.data() + range.end )
		};
	}
#endif
}

// maps embeddings easily
std::vector<std::vector<float>> VALL_E_API map_embeddings( const std::vector<llama_token>& tokens, int n_embd, const float* embds ) {
	std::vector<std::vector<float>> embedded( tokens.size() );
	for ( auto i = 0; i < tokens.size(); ++i ) {
		embedded[i].insert( embedded[i].end(), embds + (tokens[i] * n_embd), embds + ((tokens[i]+1) * n_embd) );
	}
	return embedded;
}

// handles adding either a token OR the embedding of that token into the batch
// this really, really helps avoid needing to abuse the tokenizer
void VALL_E_API batch_add( llama_batch& batch, llama_token id, int n_embd, const float* embds, llama_pos pos, bool output, const std::vector<llama_seq_id> & seq_ids ) {
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
bool VALL_E_API read_wav_from_disk(std::string in_path, std::vector<float> & audio_arr) {
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
void VALL_E_API write_wav_on_disk(std::vector<float> & audio_arr, std::string dest_path) {
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
std::vector<std::vector<int32_t>> VALL_E_API encode_audio_from_disk( struct encodec_context* ectx, const std::string& path ) {
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
std::vector<float> VALL_E_API decode_audio( struct encodec_context* ectx, const std::vector<std::vector<int32_t>>& codes_2d ) {
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

// sums embeddings over a 2D "tensor"
std::vector<std::vector<float>> VALL_E_API sum_embeddings( const std::vector<std::vector<llama_token>>& input, int n_embd, int rvq_l, const float** embds, int mode ) {
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

std::vector<float> VALL_E_API soft_max( int n_logits, const float* logits ) {
	std::vector<float> res( n_logits, 0.0f );
	float denom = 0.0f;

	for ( auto i = 0; i < n_logits; ++i ) {
		denom += exp( logits[i] );
	}
	// to-do: assert denom != 0.0f
	for ( auto i = 0; i < n_logits; ++i ) {
		res[i] = logits[i] / denom;
	}

	return res;
}

void VALL_E_API fill_batch( llama_batch& batch, input_t& input, inputs_map_t& inputs_map, int mode ) {
	// keeps track of the position for each sequence
	size_t pos = 0;
	auto n_embd = inputs_map.n_embd;

	const float* text_embds = vall_e_inputs_map_get_embeddings_p(inputs_map, "text");
	const float* rvq_l_embds = vall_e_inputs_map_get_embeddings_p(inputs_map, "rvq_l");
	const float* lang_embds = vall_e_inputs_map_get_embeddings_p(inputs_map, "lang");
	const float* task_embds = vall_e_inputs_map_get_embeddings_p(inputs_map, "task");
	const float* len_embds = vall_e_inputs_map_get_embeddings_p(inputs_map, "len");
	const float* tone_embds = vall_e_inputs_map_get_embeddings_p(inputs_map, "tone");
	const float* sep_embds = vall_e_inputs_map_get_embeddings_p(inputs_map, "sep");
	const float* prom_embds[] = {
		vall_e_inputs_map_get_embeddings_p(inputs_map, "prom|0"),
		vall_e_inputs_map_get_embeddings_p(inputs_map, "prom|1"),
		vall_e_inputs_map_get_embeddings_p(inputs_map, "prom|2"),
		vall_e_inputs_map_get_embeddings_p(inputs_map, "prom|3"),
		vall_e_inputs_map_get_embeddings_p(inputs_map, "prom|4"),
		vall_e_inputs_map_get_embeddings_p(inputs_map, "prom|5"),
		vall_e_inputs_map_get_embeddings_p(inputs_map, "prom|6"),
		vall_e_inputs_map_get_embeddings_p(inputs_map, "prom|7"),
	};
	const float* resp_embds[] = {
		vall_e_inputs_map_get_embeddings_p(inputs_map, "resps|AR:0:0"),
		vall_e_inputs_map_get_embeddings_p(inputs_map, "resps|NAR:0:1"),
		vall_e_inputs_map_get_embeddings_p(inputs_map, "resps|NAR:1:2"),
		vall_e_inputs_map_get_embeddings_p(inputs_map, "resps|NAR:2:3"),
		vall_e_inputs_map_get_embeddings_p(inputs_map, "resps|NAR:3:4"),
		vall_e_inputs_map_get_embeddings_p(inputs_map, "resps|NAR:4:5"),
		vall_e_inputs_map_get_embeddings_p(inputs_map, "resps|NAR:5:6"),
		vall_e_inputs_map_get_embeddings_p(inputs_map, "resps|NAR:6:7"),
		vall_e_inputs_map_get_embeddings_p(inputs_map, "resps|NAR:0:0"),
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
std::vector<llama_token> VALL_E_API generate( llama_context* ctx, llama_model* model, llama_sampler* smpl, input_t& input, inputs_map_t& inputs_map, int max_tokens, int mode, bool verbose ) {
	int rvq_l = input.rvq_l;
	llama_token stop_token = -1;
	int n_decode = 0; // number of tokens decoded
	int n_outputs = 0; // number of output tokens to expect
	int n_vocab = 0;
	int n_embd = 0;
	bool causal = true; // sample autoregressively or not
	const float* embds = NULL; // embeddings to map output tokens through
	ranges_t range; // I/O range

	// create batch	(targetting embeddings instead of tokens)
	llama_batch batch = llama_batch_init( CTX_SIZE, inputs_map.n_embd, CTX_SIZE );
	fill_batch( batch, input, inputs_map, mode );

	// determine how many outputs we need
	for ( auto i = 0; i < batch.n_tokens; ++i ) {
		if ( batch.logits[i] ) ++n_outputs;
	}
	if ( verbose ) printf("Prompt size: %i | Outputs: %i\n", batch.n_tokens, n_outputs);

	// bail out
	if ( n_outputs == 0 ) {
		fprintf(stderr, "%s : no tokens to decode\n", __func__);
		return {};
	}
	causal = n_outputs == 1;

	// AR mode
	std::string embd_name = "";
	if ( mode == INFERENCE_MODE_AR ) {
		embd_name = "resps|AR:0:0";
	// NAR mode
	} else if ( mode == INFERENCE_MODE_NAR ) {
		std::string k_embds[] = {
			"resps|NAR:0:0", // invalid, should never be picked
			"resps|NAR:0:1",
			"resps|NAR:1:2",
			"resps|NAR:2:3",
			"resps|NAR:3:4",
			"resps|NAR:4:5",
			"resps|NAR:5:6",
			"resps|NAR:6:7",
		};
		embd_name = k_embds[rvq_l];
	// duration inferencing mode
	} else if ( mode == INFERENCE_MODE_LEN ) {
		embd_name = "len";
	// NAR-len (demasking) inferencing mode
	} else if ( mode == INFERENCE_MODE_NAR_DEMASK ) {
		embd_name = "resps|NAR:0:0";
	}

	auto& embeddings = vall_e_inputs_map_get_embeddings(inputs_map, embd_name);
	range = embeddings.range;
	embds = embeddings.embds.data();
	n_embd = embeddings.n_embd;
	n_vocab = embeddings.n_vocab;
	stop_token = range.end - range.start - 1;

	printf("Generating in %s (%i) mode (%i:%i) (%i)\n", embd_name.c_str(), range.classifier_idx, range.start, range.end, stop_token);

	// update model's output heads / causal mode
#if LLAMA_CPP_USE_VALL_E_ARCH
	auto& userdata = *llama_get_vall_e_userdata( model );
	llama_set_output_head( model, userdata.heads[range.classifier_idx] );
#endif
	llama_set_causal_attn( ctx, causal );
	// to-do: fix GGML_ASSERT(mask->ne[0] == a->ne[0])

	std::vector<llama_token> output_tokens;
	const auto t_main_start = ggml_time_us();

	// if INFERENCE_MODE_AR || INFERENCE_MODE_LEN
	if ( causal ) {
		output_tokens.reserve(max_tokens);
		if ( verbose ) {
			printf("[");
			fflush(stdout);
		}
		while ( output_tokens.size() < max_tokens ) {
			if ( llama_decode(ctx, batch) ) {
				fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
				return output_tokens;
			}

			// ensures only tokens within our designated range are used			
		#if !LLAMA_CPP_USE_VALL_E_ARCH
			auto* logits = llama_get_logits_ith( ctx, -1 );
			for ( auto i = 0; i < inputs_map.n_vocab; ++i ) {
				if ( i < range.start || i >= range.end ) logits[i] = -INFINITY;
			}
		#endif
			// sample token
			auto t = llama_sampler_sample(smpl, ctx, -1);

			// is stop token
			if ( t == stop_token ) {
				break;
			}

			// store token
			output_tokens.emplace_back(t);
			// update batch with token
			batch_add( batch, t, inputs_map.n_embd, embds, output_tokens.size(), true );
			if ( verbose ) {
				printf("%i, ", t);
				fflush(stdout);
			}
		}
		if ( verbose ) {
			printf("]\n");
			fflush(stdout);
		}
	} else if ( mode == INFERENCE_MODE_NAR_DEMASK ) {
		// to-do: assert n_outputs == input.resp[rvq_l-1].size()
		const llama_token MASK_TOKEN = 1024; // token value for masking 
		const float PI = 3.141592653589793f;
		// to-do: derive from sampling arguments
		int32_t steps = 30; // number of demasking steps
		int32_t seq_len = n_outputs;
		float temperature = 1.5f;
		float cfg_strength = 2.5f;

		// fill with masked tokens
		output_tokens.clear();
		output_tokens.resize(n_outputs, MASK_TOKEN);

		// for CFG
		input_t null_input{};
		null_input.phn = {1, 2}; // <bos></eos>
		null_input.resp.resize(1);

		llama_batch null_batch = llama_batch_init( CTX_SIZE, inputs_map.n_embd, CTX_SIZE );
		
		// token scores to reference for masking
		std::vector<float> scores(n_outputs, 1.0);

		// do one step on many tokens
		for ( auto step = 0; step < steps; ++step ) {
			if ( verbose ) {
				printf("[%i/%i] [", step, steps);
				fflush(stdout);
			}

			float timestep = (step+1) / steps; // to-do: align with torch.linspace
			float annealing = 1.0f - timestep;
			float noise_p = cos( timestep * PI * 0.5f );
			float remask_p = 0.5f / steps;
			int32_t n_masked_tokens = std::min(int(noise_p * seq_len), 1);
			float sampling_temperature = temperature * annealing;
			float sampling_cfg_strength = timestep * cfg_strength;

			std::vector<bool> is_masked(n_outputs, false);
			std::vector<int32_t> masked_indices;
			masked_indices.reserve(n_masked_tokens);
			std::vector<float> sorted = scores;
			std::sort(sorted.begin(), sorted.end());
			masked_indices.insert( masked_indices.end(), sorted.begin(), sorted.begin() + n_masked_tokens );

			// mask off tokens
			for ( auto& idx : masked_indices ) {
				output_tokens[idx] = MASK_TOKEN;
			}
			// update token mask
			for ( auto i = 0; i < n_outputs; ++i ) {
				is_masked[i] = output_tokens[i] == MASK_TOKEN;
			}

			// update batch
			// to-do: only update the embeddings instead
			batch.n_tokens = 0;
			input.resp[0] = output_tokens;
			fill_batch( batch, input, inputs_map, mode );
			// update null batch
			null_input.resp[0] = output_tokens;
			null_batch.n_tokens = 0;
			fill_batch( null_batch, input, inputs_map, mode );

			// to-do: update sampling temperature

			// cfg decode
			if ( llama_decode(ctx, null_batch) ) {
				fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
				return output_tokens;
			}
			// copy null probabilities
			std::vector<float> null_logits(n_outputs * n_vocab, -INFINITY);
			// to-do: copy once
			for ( auto idx = 0; idx < n_outputs; ++idx ) {
				memcpy( &null_logits[idx * n_vocab], llama_get_logits_ith( ctx, null_batch.n_tokens - n_outputs + idx ), sizeof(float) * n_vocab );
			}

			// decode
			if ( llama_decode(ctx, batch) ) {
				fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
				return output_tokens;
			}
			// to-do: figure out why all logits are the same for each token......
			// "reverse" iterate from backwards indexing
			for ( auto idx = 0; idx < n_outputs; ++idx ) {
				// skip if not masked
				if ( !is_masked[idx] )
					continue;
				// ensures only tokens within our designated range are used			
				auto* logits = llama_get_logits_ith( ctx, batch.n_tokens - n_outputs + idx );
				auto* null_logit = &null_logits[idx];

			#if !LLAMA_CPP_USE_VALL_E_ARCH
				for ( auto i = 0; i < inputs_map.n_vocab; ++i ) {
					if ( i < range.start || i >= range.end ) logits[i] = -INFINITY;
				}
			#endif
				// perform softmax before modifying logits
				std::vector<float> softmaxed = soft_max( n_vocab, logits );

				// perform CFG sampling
				for ( auto i = 0; i < n_vocab; ++i ) {
					logits[i] = null_logit[i] + (logits[i] - null_logit[i]) * cfg_strength;
				}
				// sample ith token
				auto t = llama_sampler_sample(smpl, ctx, batch.n_tokens - n_outputs + idx );
				// store token if it was masked
				output_tokens[idx] = t;
				// update score if it was masked
				scores[idx] = 1.0f - softmaxed[t]; // invert so we pick the worst tokens later
				if ( verbose ) {
					printf("%i, ", t);
					fflush(stdout);
				}
			}
			if ( verbose ) {
				printf("\n");
				fflush(stdout);
			}
		}
	} else if ( mode == INFERENCE_MODE_NAR ) {
		// to-do: assert n_outputs == input.resp[rvq_l-1].size()
		output_tokens.reserve(n_outputs);
		// do one step on many tokens
		if ( llama_decode(ctx, batch) ) {
			fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
			return output_tokens;
		}
		// to-do: figure out why all logits are the same for each token......
		// "reverse" iterate from backwards indexing
		if ( verbose ) {
			printf("[");
			fflush(stdout);
		}
		for ( auto idx = 0; idx < n_outputs; ++idx ) {
			// ensures only tokens within our designated range are used			
		#if !LLAMA_CPP_USE_VALL_E_ARCH
			auto* logits = llama_get_logits_ith( ctx, batch.n_tokens - n_outputs + idx  );
			for ( auto i = 0; i < inputs_map.n_vocab; ++i ) {
				if ( i < range.start || i >= range.end ) logits[i] = -INFINITY;
			}
		#endif

			// sample ith token
			auto t = llama_sampler_sample(smpl, ctx, batch.n_tokens - n_outputs + idx);

			// store token
			output_tokens.emplace_back(t);
			if ( verbose ) {
				printf("%i, ", t);
				fflush(stdout);
			}
		}
		if ( verbose ) {
			printf("]\n");
			fflush(stdout);
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

int main( int argc, char** argv ) {
	// to-do: replace all of this with proper loading code
	int32_t ngl = 0;
	int modality = MODALITY_NAR_LEN;
	input_t input{};
	inputs_map_t inputs_map{};

	// input.phonemes = "hˈɛloː ʋˈɔrlt";
	input.phn = {1,22,111,100,4,37,115,169,11,2}; // <bos>hˈɛloː ʋˈɔrlt</eos>
	input.prom = {};
	input.resp = {};

	std::string vall_e_model_path = "./data/vall_e.gguf";
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

	llama_sampler_chain_add(smpl_ar, llama_sampler_init_top_k(0));
    llama_sampler_chain_add(smpl_ar, llama_sampler_init_top_p(1.0, 1));
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
	if ( input.prom.empty() ) {
		input.prom = encode_audio_from_disk(ectx, input_prompt_path);
	}
	//input.resp = encode_audio_from_disk(ectx, output_response_path);

	// prepare batch
	auto n_embd = llama_n_embd( model );
	auto n_vocab = llama_n_vocab( model );

	// grab input embeddings	
	vall_e_inputs_map_init( inputs_map, model );

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
		int len = 290;
		if ( !len ) {
			input.task = "len";
			output_tokens = generate( ctx, model, smpl_nar, input, inputs_map, 5, INFERENCE_MODE_LEN );
			{
				int digit = 1;
				for (int i = output_tokens.size() - 1; i >= 0; i--) {
					len += output_tokens[i] * digit;
					digit *= 10;
				}
			}
			// cap for now
			if ( len <= 0 || len > MAX_DURATION ) len = MAX_DURATION;
		}

		// fill with mask tokens
		input.resp.resize(1);
		for ( auto i = 0; i < len; ++i ) {
			input.resp[0].emplace_back( 1024 ); // fill with masked tokens
		}

		// inference NAR-len 0
		input.task = "tts";
		for ( auto l = 0; l < 8; ++l ) {
			input.rvq_l = l;
			output_tokens = generate( ctx, model, smpl_nar, input, inputs_map, 5, l == 0 ? INFERENCE_MODE_NAR_DEMASK  : INFERENCE_MODE_NAR );
			input.resp.emplace_back( output_tokens );
		}
	// AR+NAR
	} else if ( modality == MODALITY_AR_NAR ){
		input.task = "tts";
		for ( auto l = 0; l < 8; ++l ) {
			input.rvq_l = l;
			output_tokens = generate( ctx, model, l == 0 ? smpl_ar : smpl_nar, input, inputs_map, l == 0 ? MAX_DURATION : 1, l == 0 ? INFERENCE_MODE_AR  : INFERENCE_MODE_NAR );
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