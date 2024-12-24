#define DR_WAV_IMPLEMENTATION
#include "vall_e.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

io_t io_ranges[] = {
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

	{ "resps|AR:0:0", 8484, 9509, 0  }, 
	{ "resps|NAR:0:1", 9509, 10533, 1  }, 
	{ "resps|NAR:1:2", 10533, 11557, 2 }, 
	{ "resps|NAR:2:3", 11557, 12581, 3 }, 
	{ "resps|NAR:3:4", 12581, 13605, 4 }, 
	{ "resps|NAR:4:5", 13605, 14629, 5 }, 
	{ "resps|NAR:5:6", 14629, 15653, 6 }, 
	{ "resps|NAR:6:7", 15653, 16677, 7 }, 
	{ "resps|NAR:0:0", 16677, 17702, 8 }, 
};

std::vector<float> VALL_E_API read_2d_tensor( struct ggml_tensor* tensor ) {
	size_t size = tensor->ne[0] * tensor->ne[1];
	std::vector<float> res( size );
	
	auto* type_trait = ggml_get_type_traits(tensor->type);
	if ( type_trait->to_float ) {
		type_trait->to_float(tensor->data, res.data(), res.size());
	} else {
		memcpy( res.data(), tensor->data, res.size() * sizeof(float) );
	}

	return res;
}

ggml_tensor* VALL_E_API view_2d_tensor( struct ggml_tensor* tensor, int32_t start, int32_t end, int32_t dim ) {
	// to-do: implement other dim
	if ( start < 0 ) start = tensor->ne[1] + start;
	if ( end < 0 ) end = tensor->ne[1] + end;
	
	ggml_tensor* res = new ggml_tensor();
	memcpy( res, tensor, sizeof(ggml_tensor) );

	res->op     = GGML_OP_VIEW;
	res->src[0] = tensor;

	res->data   += res->nb[1] * start;
	res->ne[1]  = end - start;

	for (int i = 2; i < GGML_MAX_DIMS; i++) {
		res->nb[i] = res->nb[i - 1] * res->ne[i - 1];
	}

	return res;
}
ggml_tensor* VALL_E_API view_2d_tensor( struct ggml_context* ctx, struct ggml_tensor* tensor, int32_t start, int32_t end, int32_t dim ) {
	// to-do: implement other dim
	if ( start < 0 ) start = tensor->ne[1] + start;
	if ( end < 0 ) end = tensor->ne[1] + end;

	ggml_tensor* res = ggml_view_2d( ctx, tensor, tensor->ne[0], end - start, tensor->nb[1], tensor->nb[1] * start );

	return res;
}

void VALL_E_API print_tokens( const std::vector<llama_token>& tokens, const std::string& prefix ) {
	printf("%s[", prefix.c_str());
	for ( auto i = 0; i < tokens.size(); ++i ) {
		printf("%i%s", tokens[i], i + 1 < tokens.size() ? ", " : "");
	}
	printf("]\n");
}

const io_t& VALL_E_API vall_e_inputs_map_get( io_map_t& io_map, const std::string& name ) {
	return io_map.io[name];
}
const float* VALL_E_API vall_e_inputs_map_get_embeddings_p( io_map_t& io_map, const std::string& name ) {
	return io_map.io[name].embds.data();	
}

int32_t VALL_E_API vall_e_inputs_map_get_classifier_idx( io_map_t& io_map, const std::string& name ) {
	return io_map.io[name].head_idx;
}

void VALL_E_API vall_e_inputs_map_init( io_map_t& io_map, llama_model* model ) {
	auto n_embd = llama_n_embd( model );
	auto n_vocab = llama_n_vocab( model );
	
	io_map.n_embd = n_embd;
	io_map.n_vocab = n_vocab;

	int32_t ctx_size = 24 * 2 * ggml_tensor_overhead(); // 24 embeddings + 24 output heads (generous) (should only really need to do this for output heads since we manually handle embeddings)
	struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    io_map.ctx = ggml_init(params);

// to-do: figure a nicer way to do this
#if LLAMA_CPP_USE_VALL_E_ARCH
	auto& userdata = *llama_get_vall_e_userdata( model );

	for ( auto& entry : io_ranges ) {
		io_map.io[entry.name] = entry;

		io_map.io[entry.name].n_embd = n_embd;
		io_map.io[entry.name].n_vocab = entry.end - entry.start;
		io_map.io[entry.name].start = 0;
		io_map.io[entry.name].end = 0;
		io_map.io[entry.name].head = entry.head_idx < 0 ? NULL : userdata.heads[entry.head_idx];
	}

	io_map.io["text"].embds = read_2d_tensor(userdata.aux_embds[0]);
	io_map.io["rvq_l"].embds = read_2d_tensor(userdata.aux_embds[1]);
	io_map.io["lang"].embds = read_2d_tensor(userdata.aux_embds[2]);
	io_map.io["task"].embds = read_2d_tensor(userdata.aux_embds[3]);
	io_map.io["len"].embds = read_2d_tensor(userdata.aux_embds[4]);
	io_map.io["tone"].embds = read_2d_tensor(userdata.aux_embds[5]);
	io_map.io["sep"].embds = read_2d_tensor(userdata.aux_embds[6]);

	io_map.io["prom|0"].embds = read_2d_tensor(userdata.prom_embds[0]);
	io_map.io["prom|1"].embds = read_2d_tensor(userdata.prom_embds[1]);
	io_map.io["prom|2"].embds = read_2d_tensor(userdata.prom_embds[2]);
	io_map.io["prom|3"].embds = read_2d_tensor(userdata.prom_embds[3]);
	io_map.io["prom|4"].embds = read_2d_tensor(userdata.prom_embds[4]);
	io_map.io["prom|5"].embds = read_2d_tensor(userdata.prom_embds[5]);
	io_map.io["prom|6"].embds = read_2d_tensor(userdata.prom_embds[6]);
	io_map.io["prom|7"].embds = read_2d_tensor(userdata.prom_embds[7]);
		
	io_map.io["resps|AR:0:0"].embds = read_2d_tensor(userdata.resp_embds[0]);
	io_map.io["resps|NAR:0:1"].embds = read_2d_tensor(userdata.resp_embds[1]);
	io_map.io["resps|NAR:1:2"].embds = read_2d_tensor(userdata.resp_embds[2]);
	io_map.io["resps|NAR:2:3"].embds = read_2d_tensor(userdata.resp_embds[3]);
	io_map.io["resps|NAR:3:4"].embds = read_2d_tensor(userdata.resp_embds[4]);
	io_map.io["resps|NAR:4:5"].embds = read_2d_tensor(userdata.resp_embds[5]);
	io_map.io["resps|NAR:5:6"].embds = read_2d_tensor(userdata.resp_embds[6]);
	io_map.io["resps|NAR:6:7"].embds = read_2d_tensor(userdata.resp_embds[7]);
	io_map.io["resps|NAR:0:0"].embds = read_2d_tensor(userdata.resp_embds[8]);
#else
	auto* embds = llama_get_embedding_weights( model );
	auto* heads = llama_get_output_head_tensor( model );

	// prepare slices
	for ( auto& entry : io_ranges ) {
		io_map.io[entry.name] = entry;

		io_map.io[entry.name].n_embd = n_embd;
		io_map.io[entry.name].n_vocab = entry.end - entry.start;
		io_map.io[entry.name].embds = read_2d_tensor(view_2d_tensor( io_map.ctx, embds, entry.start, entry.end ));
		io_map.io[entry.name].head = entry.head_idx < 0 ? NULL : view_2d_tensor( io_map.ctx, heads, entry.start, entry.end );	
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
		if ( id < 0 ) for ( auto i = 0; i < n_embd; ++i ) batch.embd[batch.n_tokens * n_embd + i] = embds[i];
		else for ( auto i = 0; i < n_embd; ++i ) batch.embd[batch.n_tokens * n_embd + i] = embds[id * n_embd + i];
	// insert token (never gets used here)
	} else {
		batch.token[batch.n_tokens] = id;
	}

	batch.pos[batch.n_tokens] = pos;
	
	batch.n_seq_id[batch.n_tokens] = seq_ids.size();
	for (size_t i = 0; i < seq_ids.size(); ++i) batch.seq_id[batch.n_tokens][i] = seq_ids[i];
	batch.logits[batch.n_tokens] = output ? 1 : 0;
	
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
        fprintf(stderr, "%s: error during reading wav file\n", __func__);
        return {};
    }

    // compress audio
    if (!encodec_compress_audio(ectx, wavform.data(), wavform.size(), 1)) {
        fprintf(stderr, "%s: error during compression \n", __func__);
        return {};
    }

    int32_t* codes_data = encodec_get_codes( ectx );
    int n_codes = encodec_get_codes_size( ectx );
    int n_codebooks = 8;
    int n_frames = n_codes / n_codebooks;
    
    std::vector<std::vector<int32_t>> res(n_codebooks);

    for ( auto l = 0; l < n_codebooks; ++l ) {
    	res[l].insert( res[l].end(), codes_data + (l * n_frames), codes_data + ((l+1) * n_frames) );
    }

    return res;
}
// decodes a 2D codebook into a waveform
std::vector<float> VALL_E_API decode_audio( struct encodec_context* ectx, const std::vector<std::vector<int32_t>>& codes ) {
    int n_codebooks = codes.size();
    int n_frames = codes[0].size();
	

	std::vector<int32_t> res;
	res.reserve(n_frames * n_codebooks);
	for ( auto l = 0; l < n_codebooks; ++l ) {
		print_tokens( codes[l] );
		res.insert( res.end(), codes[l].begin(), codes[l].end() );
	}

    // decompress audio
    if (!encodec_decompress_audio(ectx, res.data(), res.size(), N_THREADS)) {
        fprintf(stderr, "%s: error during decompression\n", __func__);
        return {};
    }

    // write reconstructed audio on disk
    const float* audio_data = encodec_get_audio(ectx);
    const int audio_size = encodec_get_audio_size(ectx);
    return std::vector<float>(audio_data, audio_data + audio_size);
}

// sums embeddings over a 2D "tensor"
std::vector<std::vector<float>> VALL_E_API sum_embeddings( const std::vector<std::vector<llama_token>>& input, int n_embd, int rvq_l, const float** embds, int mode ) {
	auto n_tokens = input[0].size();
	//auto n_embd = input[0].size();

	std::vector<std::vector<float>> res( n_tokens, std::vector<float>( n_embd, 0.0 ) );

	// iterate through rvq levels (only up to inclusive the target rvq level)
	for ( auto l = 0; l < input.size() && l <= rvq_l; ++l ) {
		int offset = 0;
		// handles the cringe logic I have
		if ( mode == EMBEDDING_MODE_RESP_AR_NAR ) {
			offset = input.size() == 1 ? 0 : 1;
		} else if ( mode == EMBEDDING_MODE_RESP_NAR_LEN ) {
			offset = input.size() == 1 ? 8 : 1;
		}
		// embed the current level's tokens
		auto embedded = map_embeddings( input[l], n_embd, embds[l + offset] );

		for ( auto idx = 0; idx < n_tokens; ++idx ) {
			for ( auto embd_idx = 0; embd_idx < n_embd; ++embd_idx ) {
				res[idx][embd_idx] += embedded[idx][embd_idx];
			}
		}
	}
	return res;
}

std::vector<float> VALL_E_API soft_max( int n_logits, const float* logits ) {
	std::vector<float> res( n_logits, 0.0f );
	std::vector<float> expd( n_logits, 0.0f );
	float denom = 0.0f;

	for ( auto i = 0; i < n_logits; ++i ) {
		expd[i] = expf( logits[i] );
		denom += expd[i];
	}
	// to-do: assert denom != 0.0f
	for ( auto i = 0; i < n_logits; ++i ) {
		res[i] = expd[i] / denom;
	}

	return res;
}

std::vector<float> VALL_E_API log_soft_max( int n_logits, const float* logits ) {
	std::vector<float> res( n_logits, 0.0f );
	float denom = 0.0f;

	for ( auto i = 0; i < n_logits; ++i ) {
		denom += logits[i];
	}
	// to-do: assert denom != 0.0f
	for ( auto i = 0; i < n_logits; ++i ) {
		res[i] = logits[i] / denom;
	}

	return res;
}

void VALL_E_API fill_batch( llama_batch& batch, input_t& input, io_map_t& io_map, int mode ) {
	// keeps track of the position for each sequence
	size_t pos = 0;
	auto n_embd = io_map.n_embd;

	const float* text_embds = vall_e_inputs_map_get_embeddings_p(io_map, "text");
	const float* rvq_l_embds = vall_e_inputs_map_get_embeddings_p(io_map, "rvq_l");
	const float* lang_embds = vall_e_inputs_map_get_embeddings_p(io_map, "lang");
	const float* task_embds = vall_e_inputs_map_get_embeddings_p(io_map, "task");
	const float* len_embds = vall_e_inputs_map_get_embeddings_p(io_map, "len");
	const float* tone_embds = vall_e_inputs_map_get_embeddings_p(io_map, "tone");
	const float* sep_embds = vall_e_inputs_map_get_embeddings_p(io_map, "sep");
	const float* prom_embds[] = {
		vall_e_inputs_map_get_embeddings_p(io_map, "prom|0"),
		vall_e_inputs_map_get_embeddings_p(io_map, "prom|1"),
		vall_e_inputs_map_get_embeddings_p(io_map, "prom|2"),
		vall_e_inputs_map_get_embeddings_p(io_map, "prom|3"),
		vall_e_inputs_map_get_embeddings_p(io_map, "prom|4"),
		vall_e_inputs_map_get_embeddings_p(io_map, "prom|5"),
		vall_e_inputs_map_get_embeddings_p(io_map, "prom|6"),
		vall_e_inputs_map_get_embeddings_p(io_map, "prom|7"),
	};
	const float* resp_embds[] = {
		vall_e_inputs_map_get_embeddings_p(io_map, "resps|AR:0:0"),
		vall_e_inputs_map_get_embeddings_p(io_map, "resps|NAR:0:1"),
		vall_e_inputs_map_get_embeddings_p(io_map, "resps|NAR:1:2"),
		vall_e_inputs_map_get_embeddings_p(io_map, "resps|NAR:2:3"),
		vall_e_inputs_map_get_embeddings_p(io_map, "resps|NAR:3:4"),
		vall_e_inputs_map_get_embeddings_p(io_map, "resps|NAR:4:5"),
		vall_e_inputs_map_get_embeddings_p(io_map, "resps|NAR:5:6"),
		vall_e_inputs_map_get_embeddings_p(io_map, "resps|NAR:6:7"),
		vall_e_inputs_map_get_embeddings_p(io_map, "resps|NAR:0:0"),
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
		batch_add( batch, -1, n_embd, summed_proms_embds[i].data(), pos++, false );
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
std::vector<llama_token> VALL_E_API generate( llama_context* ctx, llama_model* model, input_t& input, io_map_t& io_map, int max_tokens, int mode, bool verbose ) {
	bool causal = true; // sample autoregressively or not
	int n_outputs = 0; // number of output tokens to expect

	// create batch	(targetting embeddings instead of tokens)
	llama_batch batch = llama_batch_init( CTX_SIZE, io_map.n_embd, CTX_SIZE );
	fill_batch( batch, input, io_map, mode );

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
		embd_name = k_embds[input.rvq_l];
	// duration inferencing mode
	} else if ( mode == INFERENCE_MODE_LEN ) {
		embd_name = "len";
	// NAR-len (demasking) inferencing mode
	} else if ( mode == INFERENCE_MODE_NAR_DEMASK ) {
		embd_name = "resps|NAR:0:0";
	}

	auto& io = vall_e_inputs_map_get(io_map, embd_name);
	const float* embds = io.embds.data();

	int32_t n_embd = io.n_embd;
	int32_t n_vocab = io.n_vocab;
	llama_token stop_token = io.end - io.start - 1;

	if ( verbose ) printf("Generating in %s (%i) mode (%i:%i) (%i)\n", embd_name.c_str(), io.head_idx, io.start, io.end, stop_token);

	// update model's output heads / causal mode
	llama_set_output_head( model, io.head );
	// to-do: figure this out......
	{
		llama_set_causal_attn( ctx, causal ); // to-do: fix GGML_ASSERT(mask->ne[0] == a->ne[0])
    //	*const_cast<bool*>(&model->hparams.causal_attn) = true; // force set this
	}

	std::vector<llama_token> output_tokens;
	const auto t_main_start = ggml_time_us();

	// if INFERENCE_MODE_AR || INFERENCE_MODE_LEN
	if ( causal ) {
		auto sparams = llama_sampler_chain_default_params();
		sparams.no_perf = false;
		llama_sampler * smpl = llama_sampler_chain_init(sparams);

		llama_sampler_chain_add(smpl, llama_sampler_init_top_k(0));
		llama_sampler_chain_add(smpl, llama_sampler_init_top_p(1.0, 1));
		llama_sampler_chain_add(smpl, llama_sampler_init_temp (1.0));
		llama_sampler_chain_add(smpl, llama_sampler_init_dist (LLAMA_DEFAULT_SEED));

		output_tokens.reserve(max_tokens);
		while ( output_tokens.size() < max_tokens ) {
			if ( llama_decode(ctx, batch) ) {
				fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
				return output_tokens;
			}
			llama_kv_cache_clear(ctx); // necessary for many reasons

			// sample token
			auto t = llama_sampler_sample(smpl, ctx, -1);

			// is stop token
			if ( t == stop_token ) {
				break;
			}

			// store token
			output_tokens.emplace_back(t);
			// update batch with token
			batch_add( batch, t, io_map.n_embd, embds, output_tokens.size(), true );
			
			if ( verbose ) print_tokens( output_tokens );
		}

		llama_sampler_free(smpl);
	} else if ( mode == INFERENCE_MODE_NAR_DEMASK ) {
		// to-do: assert n_outputs == input.resp[rvq_l-1].size()
		const llama_token MASK_TOKEN = 1024; // token value for masking 
		const float PI = 3.141592653589793f;
		// to-do: derive from sampling arguments
		int32_t steps = 10; // number of demasking steps
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

		llama_batch null_batch = llama_batch_init( CTX_SIZE, io_map.n_embd, CTX_SIZE );
		
		// token scores to reference for masking
		std::vector<float> scores(n_outputs, 1.0);

		// do one step on many tokens
		for ( auto step = 0; step < steps; ++step ) {
			float timestep = ((float)step) / steps; // to-do: align with torch.linspace
			
			float annealing = 1.0f - timestep;
			
			float sampling_temperature = temperature * annealing;
			float sampling_cfg_strength = timestep * cfg_strength;

			float noise_p = cos( timestep * PI * 0.5f );
			float remask_p = 0.5f / steps;
			
			int32_t n_masked_tokens = (noise_p + remask_p) * seq_len;
			if ( n_masked_tokens < 1 ) {
				n_masked_tokens = 1;
			}
			if ( n_masked_tokens > (n_outputs - step) ) {
				n_masked_tokens = (n_outputs - step);
			}

			// masked mask
			std::vector<bool> is_masked(n_outputs, false);
			// sort previous scores
			std::vector<score_t> sorted_scores( n_outputs );
			for ( auto i = 0; i < n_outputs; ++i ) sorted_scores[i] = { i, scores[i] };
			std::sort(sorted_scores.begin(), sorted_scores.end());
			std::reverse(sorted_scores.begin(), sorted_scores.end());
			
			// and top-k pick the worst scores
			for ( auto i = 0; i < n_masked_tokens; ++i ) {
				auto idx = sorted_scores[i].idx;

				output_tokens[idx] = MASK_TOKEN;
				is_masked[idx] = true;
			}

			if ( verbose ) print_tokens( output_tokens, "Masked tokens: " );

			// update batch
			// to-do: only update the embeddings instead
			batch.n_tokens = 0;
			input.resp[0] = output_tokens;
			fill_batch( batch, input, io_map, mode );
			// update null batch
			null_input.resp[0] = output_tokens;
			null_batch.n_tokens = 0;
			fill_batch( null_batch, input, io_map, mode );

			// cfg decode
			if ( llama_decode(ctx, null_batch) ) {
				fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
				return output_tokens;
			}
			llama_kv_cache_clear(ctx); // necessary for many reasons
			// copy null probabilities
			std::vector<float> null_logits(n_outputs * n_vocab, 0.0f);
			memcpy( null_logits.data(), llama_get_logits( ctx ), sizeof(float) * n_vocab * n_outputs );

			// decode
			if ( llama_decode(ctx, batch) ) {
				fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
				return output_tokens;
			}
			llama_kv_cache_clear(ctx); // necessary for many reasons
			
			auto sparams = llama_sampler_chain_default_params();
			sparams.no_perf = false;
			llama_sampler * smpl = llama_sampler_chain_init(sparams);

			llama_sampler_chain_add(smpl, llama_sampler_init_top_k(20));
			llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9, 1));
			llama_sampler_chain_add(smpl, llama_sampler_init_temp (sampling_temperature));
			llama_sampler_chain_add(smpl, llama_sampler_init_dist (LLAMA_DEFAULT_SEED));

			auto* logits = llama_get_logits( ctx );

			/*
			// perform CFG sampling
			for ( auto i = 0; i < n_vocab * n_outputs; ++i ) {
				logits[i] = null_logit[i] + (logits[i] - null_logit[i]) * cfg_strength;
			}
			*/

			for ( auto idx = 0; idx < n_outputs; ++idx ) {
				// skip if not masked
				if ( !is_masked[idx] ) {
					continue;
				}

				auto* logit = &logits[idx * n_vocab];
				auto* null_logit = &null_logits[idx * n_vocab];

				// perform softmax before modifying logits
				std::vector<float> softmaxed = soft_max( n_vocab, logit );

				for ( auto i = 0; i < n_vocab; ++i ) {
					logit[i] = null_logit[i] + (logit[i] - null_logit[i]) * cfg_strength;
				}

				// sample ith token
				auto t = llama_sampler_sample(smpl, ctx, batch.n_tokens - n_outputs + idx );
				// store token if it was masked
				output_tokens[idx] = t;
				// update score if it was masked
				scores[idx] = 1.0f - softmaxed[t]; // invert so we pick the worst tokens later
			}

			llama_sampler_free(smpl);

			if ( verbose ) print_tokens( output_tokens );
		}
	} else if ( mode == INFERENCE_MODE_NAR ) {
		// to-do: assert n_outputs == input.resp[rvq_l-1].size()
		output_tokens.clear();
		output_tokens.resize(n_outputs);
		// do one step on many tokens
		if ( llama_decode(ctx, batch) ) {
			fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
			return output_tokens;
		}
		llama_kv_cache_clear(ctx); // necessary for many reasons

		auto sparams = llama_sampler_chain_default_params();
		sparams.no_perf = false;
		llama_sampler * smpl = llama_sampler_chain_init(sparams);

		llama_sampler_chain_add(smpl, llama_sampler_init_top_k(20));
		llama_sampler_chain_add(smpl, llama_sampler_init_top_p(1.0, 1));
		llama_sampler_chain_add(smpl, llama_sampler_init_temp (1.0));
		llama_sampler_chain_add(smpl, llama_sampler_init_dist (LLAMA_DEFAULT_SEED));

		for ( auto idx = 0; idx < n_outputs; ++idx ) {
			// sample ith token
			auto t = llama_sampler_sample(smpl, ctx, batch.n_tokens - n_outputs + idx);

			// store token
			output_tokens[idx] = t;
		}
		if ( verbose ) print_tokens( output_tokens );
		
		llama_sampler_free(smpl);
	}

	const auto t_main_end = ggml_time_us();

	if ( verbose ) {
		printf("\n");
		fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
				__func__, output_tokens.size(), (t_main_end - t_main_start) / 1000000.0f, output_tokens.size() / ((t_main_end - t_main_start) / 1000000.0f));

		fprintf(stderr, "\n");
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
	io_map_t io_map{};

	// input.phonemes = "hˈɛloː ʋˈɔrlt";
	input.phn = {1,22,111,100,4,37,115,169,11,2}; // <bos>hˈɛloː ʋˈɔrlt</eos>
	input.prom = {
		{62,835,835,835,339,395,798,537,537,537,537,222,76,989,548,65,705,375,261,375,297,503,529,571,707,346,266,862,148,496,574,115,115,438,934,339,865,876,63,40,779,461,602,794,10,220,507,869,639,705,869,917,705,893,917,705,869,938,439,175,139,506,375,529,297,705,651,238,962,461,195,441,377,581,473,795,644,626,459,981,767,670,696,73,779,257,738,1017,1019,133,133,1017,835,604,699,626,67,92,707,92,179,179,772,869,441,799,630,238,745,904,904,904,106,133,133,1017,1017,395,883,87,519,594,1002,682,996,540,186,855,430,202,347,889,61,92,542,297,67,669,571,707,346,67,359,571,707,669,604,395,1008,810,35,621,67,600,333,123,284,568,817,243,778,464,638,610,359,538,464,975,321,700,377,484,179,284,284,621,538,464,745,171,171,159,744,744,287,461,69,15,529,67,92,669,464,515,605,24,822,865,293,865,172,638,359,562,138,839,846,775,556,775,1006,917,346,312,148,331,496,646,67,314,15,705,131,855,662,287,172,85,107,519,374,450,391,609,643,778,80,287,794,794,115,785,794,461,699,519,932,522,652,262,508,902,932,932,391,769,18,507,90,442,762,610,610,669,605,35,855,56,989,863,195,464,604,257,904,632,786,951,461,239,195,878,771,146,481,146,481,434,643,917,280,67,464,115,744,744,115,115,115,819,709,63,907,359,519,996,616,682,996,616,519,762,917,841,772,568,954,600,422,893,592,464,626,86,143,615,171,744,744,196,115,821,415,521,799,654,839,644,473,592,953,523,855,738,855,855,876,1017,63,329},
		{913,859,740,740,937,601,961,961,877,747,747,559,474,618,20,316,58,316,180,112,290,869,610,869,869,943,127,153,236,794,282,857,984,196,875,648,993,913,860,616,38,833,620,133,123,992,247,367,252,50,298,27,27,631,163,784,271,20,843,514,869,258,180,66,803,281,123,493,831,102,556,992,385,122,31,251,990,827,26,347,460,43,43,460,228,43,841,913,302,544,544,277,859,404,646,775,315,848,726,185,203,314,203,174,252,174,378,954,214,993,924,809,277,765,363,544,363,518,791,185,454,193,193,193,193,193,573,977,924,76,434,56,193,962,610,24,954,459,396,112,903,137,398,474,506,791,839,399,102,25,205,792,459,474,526,817,869,192,792,593,878,506,24,410,539,788,522,667,566,584,588,992,444,24,869,925,635,393,903,742,320,1023,833,136,216,924,220,24,563,630,968,96,708,24,708,127,399,364,67,740,381,981,203,248,607,744,252,996,474,582,248,527,423,25,387,94,229,775,122,474,792,367,650,371,413,448,448,784,506,795,848,298,27,526,96,905,70,693,956,1002,1002,37,747,857,993,124,193,193,193,193,732,732,732,992,447,792,929,291,289,524,451,27,27,524,202,693,374,1002,125,732,585,367,317,679,395,413,189,493,386,650,110,912,505,384,399,851,367,367,27,230,988,810,975,842,956,1002,4,551,729,956,1002,750,648,231,950,193,96,912,410,732,539,103,193,904,491,213,792,792,998,193,399,151,410,96,673,497,1002,241,833,956,630,43,399,775,732,792,792,792,792,917,750,185,812,812,700,859,841,363,833,630},
		{786,36,821,937,1000,705,1016,345,345,470,165,581,95,404,95,95,1006,477,95,95,691,254,997,657,176,124,95,673,489,326,218,437,907,590,752,541,1016,821,445,563,181,555,181,345,576,190,987,0,265,997,488,12,598,687,152,108,52,95,95,71,87,945,95,997,754,488,955,694,925,82,18,1020,1006,542,788,441,325,532,246,132,560,532,947,655,653,842,732,36,36,829,36,937,989,989,752,651,87,489,677,260,789,462,95,227,986,955,95,810,624,435,280,868,832,879,863,821,829,937,168,270,489,544,909,562,957,0,593,714,675,690,626,227,794,489,489,563,489,298,269,741,249,516,360,240,516,336,93,808,1022,682,555,737,147,405,476,895,323,694,412,689,963,72,193,298,181,521,741,193,93,153,773,677,689,495,30,564,719,1020,559,940,53,53,53,929,360,971,403,1012,997,919,957,433,919,787,401,401,355,276,370,414,690,697,330,629,552,930,720,259,579,221,62,945,135,1020,626,663,401,153,997,381,830,185,587,853,207,126,66,529,410,113,997,488,431,563,488,488,719,746,790,296,843,752,790,23,984,292,41,27,120,249,124,900,358,801,227,978,95,997,997,997,371,561,86,388,52,667,601,894,545,997,498,900,494,365,852,986,95,841,664,256,18,1020,963,901,447,498,262,388,691,997,646,651,757,468,114,601,437,940,212,655,541,970,870,521,237,957,563,794,563,564,620,489,351,489,489,257,733,629,489,227,622,962,7,598,374,470,114,159,211,298,363,843,818,153,59,452,529,258,419,605,689,526,39,982,829,982,752,678,1005,312},
		{673,673,919,866,762,961,52,674,528,528,675,526,12,753,297,967,661,845,482,303,338,1021,506,445,247,214,206,94,434,799,210,885,552,695,853,1022,916,762,764,721,445,434,529,999,771,708,767,498,282,736,227,150,299,12,536,767,321,561,12,530,147,530,262,325,196,990,874,997,944,875,426,12,282,571,571,282,365,534,365,424,89,388,563,222,31,1019,624,74,215,651,1018,74,956,1022,74,18,633,350,72,448,454,769,267,938,12,534,929,723,829,614,505,364,1018,1014,838,673,919,74,223,761,266,78,177,736,20,718,425,1001,366,58,874,58,153,627,312,197,801,530,767,674,196,633,327,425,376,413,1019,209,594,383,744,458,468,711,282,885,640,435,655,571,556,1020,310,116,273,116,504,633,15,736,633,448,662,612,487,345,19,612,665,556,198,778,705,403,706,31,196,197,536,805,427,339,161,241,116,504,58,945,853,734,670,424,807,19,397,175,144,419,19,221,697,68,321,800,210,824,972,712,911,362,427,694,182,651,972,863,684,887,548,806,27,627,639,432,193,103,198,436,837,366,212,125,1001,493,874,808,17,17,127,204,530,300,345,425,246,240,640,906,340,310,633,246,774,114,633,522,777,874,494,577,353,939,571,693,857,722,530,521,354,492,735,214,806,483,736,530,118,234,536,177,132,522,349,259,436,973,528,414,224,762,212,854,744,271,568,127,323,736,304,499,499,78,536,736,805,232,126,468,566,611,52,339,450,258,157,602,594,854,602,599,82,124,472,563,666,174,936,818,66,758,627,52,350,999,734,215,919,1018,874,885},
		{528,448,646,190,222,884,939,907,907,673,413,786,527,517,710,449,119,531,565,762,531,501,522,246,162,871,8,594,206,937,462,712,862,151,103,261,882,990,1007,314,683,864,693,812,319,786,107,531,31,342,632,460,269,429,531,531,717,417,321,671,1015,152,467,863,285,875,941,417,475,825,596,957,117,460,162,162,117,630,735,527,272,558,38,39,605,375,39,900,862,646,712,804,622,963,407,93,828,796,306,415,70,667,371,531,1000,411,710,162,812,381,673,498,691,884,928,712,528,48,630,24,593,901,973,579,722,75,139,909,919,328,764,393,777,753,512,577,175,577,512,922,834,863,30,69,94,68,616,691,835,335,486,345,306,374,732,938,580,311,715,495,527,1008,306,369,663,512,369,320,360,80,42,1021,1021,1021,175,568,526,362,320,317,488,613,937,548,966,545,596,177,306,480,522,577,512,512,638,1008,82,100,696,89,714,531,639,460,679,718,492,509,492,624,460,572,531,306,19,473,915,558,285,319,713,1018,381,877,667,425,905,43,437,632,634,324,306,207,324,303,48,69,467,39,902,599,3,617,465,78,918,459,1009,427,751,145,531,349,356,1021,157,507,780,624,165,507,144,270,94,414,899,379,947,994,853,107,586,652,877,92,19,91,188,544,624,470,503,513,13,192,563,145,531,618,743,470,62,701,499,436,679,505,198,959,3,766,839,437,491,395,1021,512,306,512,356,851,1021,1021,78,690,856,735,286,280,4,1008,369,359,309,651,864,561,170,692,952,877,520,959,306,37,1021,31,236,162,773,522,254,446,606,691,804,882,58,974},
		{1011,939,881,881,140,937,724,724,937,1011,381,229,965,251,745,69,305,206,566,813,503,116,940,127,353,621,57,779,595,744,755,530,701,862,760,443,293,768,156,281,960,504,327,979,55,790,545,953,830,759,667,485,861,63,485,55,898,581,520,49,99,651,940,945,685,621,728,487,650,530,934,378,522,522,522,996,534,522,739,534,378,543,94,602,390,948,692,692,41,41,768,412,982,692,692,774,176,791,526,497,57,940,542,685,694,916,813,890,357,193,430,863,929,412,412,903,140,763,465,707,569,925,859,985,24,411,835,298,293,791,837,460,182,296,137,474,809,111,376,1021,111,490,111,938,542,578,477,506,57,385,300,873,240,104,667,204,515,834,24,125,113,980,111,997,859,997,376,193,490,824,511,799,719,575,451,575,251,222,630,429,920,788,300,993,641,154,816,940,618,130,940,462,823,955,1001,569,508,632,2,903,399,333,709,489,726,932,725,777,970,843,717,940,211,534,274,161,392,103,31,462,813,985,638,213,352,219,236,381,287,111,87,818,953,112,336,980,1016,72,960,426,238,60,9,487,665,129,24,24,162,312,411,111,157,473,466,222,940,341,55,457,712,179,451,111,831,918,826,814,940,30,468,240,207,389,923,186,95,300,876,679,576,543,582,111,227,312,112,545,747,378,165,158,610,601,425,238,704,630,124,644,949,982,297,868,569,24,57,465,24,859,111,24,752,775,24,647,465,495,57,24,57,227,907,296,581,843,1013,514,555,319,937,347,478,186,684,15,241,534,369,381,846,578,314,711,814,435,41,986,673,991},
		{485,748,562,562,485,380,834,997,78,963,755,142,978,135,362,421,217,79,530,1012,972,946,127,587,838,818,456,548,424,479,944,650,694,447,391,616,938,908,206,259,998,292,818,128,353,273,566,796,333,146,110,986,571,451,166,229,421,300,911,689,329,145,287,273,542,808,301,491,0,278,825,442,0,100,818,826,66,904,642,566,135,305,999,993,905,485,755,782,365,977,485,1015,570,1002,755,169,967,36,721,1019,273,931,273,166,216,31,346,946,32,290,362,828,464,748,782,1002,1015,755,1014,100,315,777,549,177,882,110,603,975,531,608,67,1011,950,465,368,416,798,941,635,602,553,300,200,644,498,325,786,734,342,222,403,1,716,175,899,273,40,333,999,74,54,644,408,976,407,631,577,338,435,612,333,273,162,709,882,555,384,995,173,459,442,72,72,200,72,711,219,282,716,442,431,801,976,130,622,72,582,384,516,772,0,440,1001,249,1,953,65,945,438,249,511,561,205,507,821,998,427,746,290,544,426,693,999,190,214,167,219,534,166,325,975,414,326,326,268,679,991,418,868,445,632,160,380,890,346,315,806,258,806,486,326,797,471,18,790,33,66,63,66,224,38,599,599,110,801,761,18,936,230,253,171,393,774,887,887,403,466,495,524,261,666,256,687,759,263,713,185,454,242,988,185,161,911,430,86,550,439,327,527,671,782,383,916,590,315,806,583,465,785,321,315,421,856,66,352,0,634,540,362,948,185,16,224,372,694,259,648,87,733,659,603,67,269,901,66,566,173,705,746,566,911,10,743,860,78,782,1002,755,389,175},
		{948,948,975,975,948,322,672,639,902,55,916,439,498,389,407,682,451,401,386,440,499,348,736,891,603,762,783,407,886,76,543,699,137,458,639,253,63,475,55,436,502,888,542,131,524,167,738,131,907,29,378,545,227,382,478,399,218,872,917,202,330,2,371,264,667,355,1016,768,590,408,463,542,214,202,715,891,840,297,509,689,290,439,672,714,528,940,1019,534,975,475,1019,835,975,558,975,981,330,635,96,858,606,627,367,191,191,669,40,873,359,267,701,426,210,1012,899,975,475,1012,610,6,300,749,231,616,877,631,720,574,551,398,503,789,684,664,390,277,150,990,823,190,971,903,175,863,316,965,988,988,800,612,336,506,242,847,389,939,415,202,83,317,2,153,365,363,57,2,891,965,300,754,763,426,555,621,303,415,367,902,829,741,119,380,902,25,884,439,822,49,76,760,566,316,249,555,774,955,834,309,859,173,935,812,682,586,141,606,197,131,644,631,913,586,202,117,810,884,76,592,754,531,586,925,649,583,145,816,821,283,871,1017,316,377,646,339,201,76,780,76,976,217,38,598,977,617,825,833,49,231,749,749,633,205,231,271,50,249,684,555,982,526,895,288,22,57,722,996,260,1018,110,833,644,738,648,468,798,297,769,282,197,402,465,510,194,930,182,909,749,986,187,187,917,38,38,985,985,988,815,878,814,459,237,768,781,649,683,749,934,729,463,181,625,231,917,96,499,839,720,439,842,205,808,338,617,681,326,446,905,346,647,533,49,728,147,432,846,536,586,611,49,879,872,893,859,859,961,989,975,701,495,65},
	};
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
	ctx_params.n_threads = N_THREADS;
	ctx_params.n_threads_batch = N_THREADS;
	ctx_params.no_perf = false;
	ctx_params.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL; 

	llama_context* ctx = llama_new_context_with_model(model, ctx_params);
	if (ctx == NULL) {
		fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
		return 1;
	}

	// initialize the sampler
    /*
	auto sparams = llama_sampler_chain_default_params();
	sparams.no_perf = false;
	llama_sampler * smpl = llama_sampler_chain_init(sparams);

	llama_sampler_chain_add(smpl, llama_sampler_init_top_k(0));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(1.0, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp (1.0));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist (LLAMA_DEFAULT_SEED));
    */

    
	struct encodec_context* ectx = encodec_load_model(encodec_model_path.c_str(), 0, ngl);
	if (!ectx) {
		fprintf(stderr, "%s: error during loading model\n", __func__);
		return 1;
	}
	
	encodec_set_target_bandwidth(ectx, 6);
	encodec_set_sample_rate(ectx, 24000);

	// load wavform
	if ( input.prom.empty() ) {
		input.prom = encode_audio_from_disk(ectx, input_prompt_path);
	}
	//input.resp = encode_audio_from_disk(ectx, output_response_path);
    

	// grab input embeddings	
	vall_e_inputs_map_init( io_map, model );

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

	// check for embds
	/*
	{
		input.task = "len";
		printf("batch init\n");
		llama_batch batch = llama_batch_init( CTX_SIZE, io_map.n_embd, CTX_SIZE );
		printf("fill init\n");
		fill_batch( batch, input, io_map, INFERENCE_MODE_LEN );
		printf("filled init\n");

		for ( auto i = 0; i < batch.n_tokens; ++i ) {
			float summed = 0;
			for ( auto j = 0; j < 1024; ++j ) {
				summed += batch.embd[i * 1024 + j];
			}
			printf("%i:  \t%i   \t%f\n", i, batch.pos[i], summed);
		}
	}
	*/

	// inference
	std::vector<llama_token> output_tokens;
	// NAR-len demasking
	if ( modality == MODALITY_NAR_LEN ) {
		// inference len
		int len = 0;
		if ( !len ) {
			input.task = "len";
			output_tokens = generate( ctx, model, input, io_map, 5, INFERENCE_MODE_LEN );
			{
				int digit = 1;
				for (auto it = output_tokens.rbegin(); it < output_tokens.rend(); ++it) {
					len += (*it) * digit;
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
		/*
		input.resp = {
			{922,738,461,341,341,10,416,416,416,416,346,346,346,346,346,484,484,484,484,484,484,333,442,442,359,359,359,459,459,975,975,626,626,626,626,626,610,359,359,359,359,359,359,359,359,359,610,610,442,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,638,638,638,638,975,975,672,875,63,144},
		//	{993,700,384,213,794,10,305,778,58,225,118,260,768,768,260,474,903,732,70,992,447,70,1000,665,848,379,485,934,181,795,438,298,688,324,934,756,395,795,110,328,343,172,768,871,593,355,396,783,24,24,911,20,27,562,697,616,668,27,27,755,20,505,248,79,822,461,197,156,27,492,151,1013,669,669,562},
		//	{626,989,936,488,511,624,997,112,112,648,210,650,563,650,41,41,490,920,977,986,920,927,131,167,167,968,346,168,167,168,120,355,766,599,712,390,558,810,948,332,332,867,994,346,955,392,920,452,576,346,52,254,52,307,897,307,968,920,167,563,167,167,167,968,167,488,968,488,1001,938,563,741,432,566,758},
		//	{916,874,798,212,496,751,620,616,982,745,975,890,890,141,141,321,321,214,899,42,151,722,310,971,774,35,627,995,27,43,248,248,595,774,942,352,810,35,384,340,654,639,89,214,737,197,657,45,622,321,337,19,483,679,938,938,682,938,938,141,938,310,114,724,116,327,372,607,607,310,204,713,762,853,853},
		};
		*/

		// inference NAR-len 0
		input.task = "tts";
		for ( auto l = 0; l < 8; ++l ) {
			input.rvq_l = l;
			output_tokens = generate( ctx, model, input, io_map, 5, l == 0 ? INFERENCE_MODE_NAR_DEMASK  : INFERENCE_MODE_NAR );
			if ( l == 0 ) input.resp.clear();
			input.resp.emplace_back( output_tokens );
		}
	// AR+NAR
	} else if ( modality == MODALITY_AR_NAR ){
		input.task = "tts";
		for ( auto l = 0; l < 8; ++l ) {
			input.rvq_l = l;
			output_tokens = generate( ctx, model, input, io_map, l == 0 ? MAX_DURATION : 1, l == 0 ? INFERENCE_MODE_AR  : INFERENCE_MODE_NAR );
			input.resp.emplace_back( output_tokens );
		}
	}

	// write audio to disk
	auto waveform = decode_audio( ectx, input.resp );
	write_wav_on_disk( waveform, output_response_path );

	// cleanup
	encodec_free(ectx);

	llama_free(ctx);
	
	llama_free_model(model);

	return 0;
}