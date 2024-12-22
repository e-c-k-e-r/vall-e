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
#include <iostream>

#include "_llama.h" // cringe hotfix but I have to do this until llama.cpp's API exposes the tok_embd

// stores the raw inputs to be fed
struct input_t {
	std::string task = "tts";

	std::vector<llama_token> phonemes = {};
	llama_token lang = 0;
	llama_token rvq_l = 0;
	std::vector<std::vector<llama_token>> prom = {};
	std::vector<std::vector<llama_token>> resp = {};
};
// handles all the cringe logic of slicing embeddings
struct embeddings_t {
	int n_embd = 0;
	int n_vocab = 0;
	float* embds = NULL;

	int text_embd_start = 0; // <unk>
	int rvq_level_embd_start = 17666; // <|RVQ:0>
	int len_embd_start = 17674; // <|len:0|>
	int lang_embd_start = 17686; // <|lang:en|>
	int task_embd_start = 17692; // <|task:tts|>
	int sep_embd_start = 17685; // <|sep|>
	int prom_embd_start[8] = {
		256 + (1024 * 0), // <|P|0:0|>
		256 + (1024 * 1), // <|P|1:0|>
		256 + (1024 * 2), // <|P|2:0|>
		256 + (1024 * 3), // <|P|3:0|>
		256 + (1024 * 4), // <|P|4:0|>
		256 + (1024 * 5), // <|P|5:0|>
		256 + (1024 * 6), // <|P|6:0|>
		256 + (1024 * 7), // <|P|7:0|>
	};
	int resp_embd_start[9] = {
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

	float* text_embds = NULL; // &embds[text_embd_start * n_embd];
	float* rvq_level_embd = NULL; // &embds[rvq_level_embd_start * n_embd];
	float* len_embd = NULL; // &embds[len_embd_start * n_embd];
	float* lang_embd = NULL; // &embds[lang_embd_start * n_embd];
	float* task_embd = NULL; // &embds[task_embd_start * n_embd];
	float* sep_embd = NULL; // &embds[sep_embd_start * n_embd];

	float* prom_embds[8] = {
		NULL, // &embds[prom_embd_start[0] * n_embd],
		NULL, // &embds[prom_embd_start[1] * n_embd],
		NULL, // &embds[prom_embd_start[2] * n_embd],
		NULL, // &embds[prom_embd_start[3] * n_embd],
		NULL, // &embds[prom_embd_start[4] * n_embd],
		NULL, // &embds[prom_embd_start[5] * n_embd],
		NULL, // &embds[prom_embd_start[6] * n_embd],
		NULL, // &embds[prom_embd_start[7] * n_embd],
	};
	float* resps_embds[9] = {
		NULL, // &embds[resp_embd_start[0] * n_embd],
		NULL, // &embds[resp_embd_start[1] * n_embd],
		NULL, // &embds[resp_embd_start[2] * n_embd],
		NULL, // &embds[resp_embd_start[3] * n_embd],
		NULL, // &embds[resp_embd_start[4] * n_embd],
		NULL, // &embds[resp_embd_start[5] * n_embd],
		NULL, // &embds[resp_embd_start[6] * n_embd],
		NULL, // &embds[resp_embd_start[7] * n_embd],
		NULL, // &embds[resp_embd_start[8] * n_embd],
	};

	embeddings_t( int n_embd = 0, int n_vocab = 0, float* embds = NULL ) {
		init( n_embd, n_vocab, embds );
	}

	void init( int n_embd, int n_vocab, float* embds = NULL ) {
		if ( !n_embd || !n_vocab || !embds ) return;

		this->n_embd = n_embd;
		this->n_vocab = n_vocab;
		this->embds = embds;

		text_embds = &embds[text_embd_start * n_embd];
		rvq_level_embd = &embds[rvq_level_embd_start * n_embd];
		len_embd = &embds[len_embd_start * n_embd];
		lang_embd = &embds[lang_embd_start * n_embd];
		task_embd = &embds[task_embd_start * n_embd];
		sep_embd = &embds[sep_embd_start * n_embd];

		for ( auto i = 0; i < 8; ++i ) prom_embds[i] = &embds[prom_embd_start[i] * n_embd];
		for ( auto i = 0; i < 9; ++i ) resps_embds[i] = &embds[resp_embd_start[i] * n_embd];
	}
};

// maps embeddings easily
std::vector<std::vector<float>> map_embeddings( const std::vector<llama_token>& tokens, int n_embd, float* embds ) {
	std::vector<std::vector<float>> embedded( tokens.size() );
	for ( auto i = 0; i < tokens.size(); ++i ) {
		embedded[i].insert( embedded[i].end(), embds + (tokens[i] * n_embd), embds + ((tokens[i]+1) * n_embd) );
	}
	return embedded;
}

// handles adding either a token OR the embedding of that token into the batch
// this really, really helps avoid needing to abuse the tokenizer
void batch_add( llama_batch& batch, llama_token id, int n_embd, float* embds, llama_pos pos, bool output, const std::vector<llama_seq_id> & seq_ids = {0} ) {
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
	// printf("[%i] Adding: %i | %i | %p | %i\n", batch.n_tokens-1, id, pos, embds, output );

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
const int EMBEDDING_MODE_RESP_AR_NAR = 0;
const int EMBEDDING_MODE_RESP_NAR_LEN = 0;

const int INFERENCE_MODE_LEN = 0;
const int INFERENCE_MODE_AR = 1;
const int INFERENCE_MODE_NAR_DEMASK = 2;
const int INFERENCE_MODE_NAR = 4;

const int MODALITY_AR_NAR = 0;
const int MODALITY_NAR_LEN = 0;

const int MAX_DURATION = 75; // * 12;

// sums embeddings over a 2D "tensor"
std::vector<std::vector<float>> sum_embeddings( const std::vector<std::vector<llama_token>>& input, int n_embd, int rvq_l, float** embds, int mode = EMBEDDING_MODE_PROM ) {
	std::vector<std::vector<float>> res( input.size() );
	res.resize( input[0].size() );
	for ( auto& e : res ) e.resize( n_embd );
	// iterate through rvq levels (only up to inclusive the target rvq level)
	for ( auto l = 0; l < input.size() && l <= rvq_l; ++l ) {
		int offset = 0;
		// handles the cringe logic I have
		if ( mode == EMBEDDING_MODE_RESP_AR_NAR ) {
			offset = input.size() == 1 ? 0 : 2;
		} else if ( mode == EMBEDDING_MODE_RESP_NAR_LEN ) {
			offset = input.size() == 1 ? 1 : 2;
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

void fill_batch( llama_batch& batch, input_t& input, embeddings_t& embeddings_map, int mode ) {
	// keeps track of the position for each sequence
	size_t pos = 0;
	auto n_embd = embeddings_map.n_embd;

	// insert text tokens
	for ( auto& id : input.phonemes ) batch_add( batch, id, n_embd, embeddings_map.text_embds, pos++, false );
	batch_add( batch, 0, n_embd, embeddings_map.sep_embd, pos++, false );
	pos = 0;
	// insert lang token
	batch_add( batch, input.lang, n_embd, embeddings_map.lang_embd, pos++, false );
	batch_add( batch, 0, n_embd, embeddings_map.sep_embd, pos++, false );
	pos = 0;
	// insert rvq level token
	batch_add( batch, input.rvq_l, n_embd, embeddings_map.rvq_level_embd, pos++, false );
	batch_add( batch, 0, n_embd, embeddings_map.sep_embd, pos++, false );
	pos = 0;
	// insert prom tokens
	auto summed_proms_embds = sum_embeddings( input.prom, n_embd, input.rvq_l, embeddings_map.prom_embds );
	for ( auto i = 0; i < summed_proms_embds.size(); ++i ) {
		batch_add( batch, -1, n_embd, &summed_proms_embds[i][0], pos++, false );
	}
	batch_add( batch, 0, n_embd, embeddings_map.sep_embd, pos++, mode == INFERENCE_MODE_AR ); // set as the last logit if AR
	pos = 0;

	// input starting len token
	if ( input.task == "len" ) {
		batch_add( batch, 0, n_embd, embeddings_map.len_embd, pos++, true );
		pos = 0;
	}

	// insert resp tokens
	if ( !input.resp.empty() ) {
		auto summed_resps_embds = sum_embeddings( input.resp, n_embd, input.rvq_l, embeddings_map.resps_embds, mode == INFERENCE_MODE_AR ? EMBEDDING_MODE_RESP_AR_NAR : EMBEDDING_MODE_RESP_NAR_LEN );
		for ( auto i = 0; i < summed_resps_embds.size(); ++i ) {
			batch_add( batch, -1, n_embd, &summed_resps_embds[i][0], pos++, true );
		}
		pos = 0;
	}
}

// generation code, should handle all modalities easily
std::vector<llama_token> generate( llama_context* ctx, llama_model* model, llama_sampler* smpl, input_t& input, embeddings_t& embeddings_map, int max_tokens, int mode, bool verbose = true ) {
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
	
	if ( verbose ) printf("Prompt size: %i | Logits: %i\n", batch.n_tokens, n_logits);

	// NAR mode, cap at one step
	if ( n_logits > 1 ) {
		max_tokens = n_logits;
	}

	if ( n_logits == 0 ) {
		fprintf(stderr, "%s : no tokens to decode\n", __func__);
		return {};
	}

	float* embds = NULL;
	int logit_range[2];
	if ( mode == INFERENCE_MODE_AR ) {
		logit_range[0] = embeddings_map.resp_embd_start[0];
		logit_range[1] = embeddings_map.resp_embd_start[1];
		
		embds = embeddings_map.resps_embds[0];
		
		stop_token = embeddings_map.resp_embd_start[1] - 1; // <|AR|0:STOP|>
	} else if ( mode == INFERENCE_MODE_NAR_DEMASK ) {
		logit_range[0] = embeddings_map.resp_embd_start[1];
		logit_range[1] = embeddings_map.resp_embd_start[2];
		
		embds = embeddings_map.resps_embds[1];

		stop_token = embeddings_map.resp_embd_start[2] - 1; // <|NAR|0:STOP|>
	} else if ( mode == INFERENCE_MODE_NAR ) {
		logit_range[0] = embeddings_map.resp_embd_start[2+rvq_l];
		logit_range[1] = embeddings_map.resp_embd_start[3+rvq_l];
		
		embds = embeddings_map.resps_embds[2];
	} else if ( mode == INFERENCE_MODE_LEN ) {
		logit_range[0] = embeddings_map.len_embd_start;
		logit_range[1] = embeddings_map.len_embd_start + 11;
		
		embds = embeddings_map.len_embd;
		
		stop_token = embeddings_map.len_embd_start + 10;
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
			for ( auto i = 0; i < embeddings_map.n_vocab; ++i ) {
				// out of target logit range, set to never happen
				if ( i < logit_range[0] || i >= logit_range[1] ) logits[i] = -INFINITY;
			}

			// sample the next token
			auto t = llama_sampler_sample(smpl, ctx, -i);

			if ( verbose ) {
				// print token for debugging
				char buf[256];
				int n = llama_token_to_piece( model, t, buf, sizeof(buf), 0, true );
				if ( n < 256 ) buf[n] = '\0';
				printf("%s\n", buf );
			}

			// is stop token
			if ( t == stop_token ) {
				max_tokens = 0;
				break;
			}

			// offset into range
			t -= logit_range[0];
	
			output_tokens.emplace_back(t);
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
	int modality = MODALITY_AR_NAR;
	input_t input{};
	embeddings_t embeddings_map{};

	input.phonemes = {1,85,4,128,26,4,186,4,89,33,25,4,48,4,134,25,52,86,4,34,97,27,11,2}; // <bos>hˈɛloː ʋˈɔrlt</eos>

	std::string vall_e_model_path = "./data/vall_e-F16.gguf";
	std::string encodec_model_path = "./data/encodec.bin";
	std::string input_prompt_path = "./data/prom.wav";
	std::string output_response_path = "./data/resp.wav";

	// load dynamic backends
	ggml_backend_load_all();

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
	ctx_params.n_ctx = 22500;
	ctx_params.n_batch = 22500;
	ctx_params.n_ubatch = 22500;
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
	// llama_sampler_chain_add(smpl_ar, llama_sampler_init_dist (1130));
	
	llama_sampler_chain_add(smpl_nar, llama_sampler_init_greedy());

	// prepare batch
	auto n_embd = llama_n_embd( model );
	auto n_vocab = llama_n_vocab( model );

	// grab input embeddings	
	std::vector<float> embds( n_embd * n_vocab );
	auto* qtype = ggml_get_type_traits(model->tok_embd->type);
	// dequantize if needed
	if ( ggml_is_quantized(model->tok_embd->type) ) {
		qtype->to_float(model->tok_embd->data, embds.data(), embds.size());
	}
	// update mapping
	embeddings_map.init( n_embd, n_vocab, embds.data() );

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
			input.resp[0].emplace_back( embeddings_map.resp_embd_start[3] - 1 ); // fill with masked tokens
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