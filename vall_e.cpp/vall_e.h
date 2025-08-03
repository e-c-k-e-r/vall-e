#pragma once

// C++ deps
#include <string>
#include <vector>
#include <unordered_map>

#include <llama.h>

// handles defining platform specific macros and import/export decorators (copied from my engine's uf/config.h)
#if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
	// Windows
	#define VALL_E_ENV "Windows"
	#define VALL_E_ENV_WINDOWS 1
	#define VALL_E_ENV_HEADER "windows.h"
	#if defined(__CYGWIN__)
		#define to_string(var) string(var)
	#endif
	#ifndef _WIN32_WINNT
		#define _WIN32_WINNT 0x0600
	#endif
	#ifndef WINVER
		#define WINVER 0x0600
	#endif
	
	#define VALL_E_IO_ROOT "./data/"
#elif defined(linux) || defined(__linux)
	// Linux
	#define VALL_E_ENV "Linux"
	#define VALL_E_ENV_LINUX 1
	#define VALL_E_ENV_HEADER "linux.h"
	
	#define VALL_E_IO_ROOT "./data/"
#elif defined(__APPLE__) || defined(MACOSX) || defined(macintosh) || defined(Macintosh)
	// MacOS
	#define VALL_E_ENV "OSX"
	#define VALL_E_ENV_OSX 1
	#define VALL_E_ENV_HEADER "osx.h"
	
	#define VALL_E_IO_ROOT "./data/"
#elif defined(__FreeBSD__) || defined(__FreeBSD_kernel__)
	// FreeBSD
	#define VALL_E_ENV "FreeBSD"
	#define VALL_E_ENV_FREEBSD 1
	#define VALL_E_ENV_HEADER "freebsd.h"
	
	#define VALL_E_IO_ROOT "./data/"
#elif defined(__sh__)
	// Dreamcast
	#define VALL_E_ENV "Dreamcast"
	#define VALL_E_ENV_DREAMCAST 1
	#define VALL_E_ENV_HEADER "dreamcast.h"
	#include VALL_E_ENV_HEADER

	#define _arch_dreamcast

	#define VALL_E_IO_ROOT "/cd/"
#else
	// Unsupported system
	#define VALL_E_ENV "Unknown"
	#define VALL_E_ENV_UNKNOWN 1
	#define VALL_E_ENV_HEADER "unknown.h"
	#warning Using "unknown"
 	#error No support
#endif

#if !defined(VALL_E_STATIC)
	#if defined(VALL_E_ENV_WINDOWS)
		// Windows compilers need specific (and different) keywords for export and import
		#define VALL_E_API_EXPORT __declspec(dllexport)
		#define VALL_E_API_IMPORT __declspec(dllimport)
		// For Visual C++ compilers, we also need to turn off this annoying C4251 warning
		#ifdef _MSC_VER
			#pragma warning(disable : 4251)
		#endif
	#else // Linux, FreeBSD, Mac OS X
		#if __GNUC__ >= 4
			// GCC 4 has special keywords for showing/hidding symbols,
			// the same keyword is used for both importing and exporting
			#define VALL_E_API_EXPORT __attribute__ ((__visibility__ ("default")))
			#define VALL_E_API_IMPORT __attribute__ ((__visibility__ ("default")))
		#else
			// GCC < 4 has no mechanism to explicitely hide symbols, everything's exported
			#define VALL_E_API_EXPORT
			#define VALL_E_API_IMPORT
		#endif
	#endif
#else
	// Static build doesn't need import/export macros
	#define VALL_E_API_EXPORT
	#define VALL_E_API_IMPORT
#endif

#ifdef VALL_E_EXPORTS
	#define VALL_E_API VALL_E_API_EXPORT
#else
	#define VALL_E_API VALL_E_API_IMPORT
#endif

typedef llama_token token_t;
typedef std::vector<std::vector<token_t>> vall_e_audio_codes_t;

const int ENCODEC_FRAMES_PER_SECOND = 75;
const int MAX_DURATION = ENCODEC_FRAMES_PER_SECOND * 12;
const int CTX_SIZE = 2048;
const int N_THREADS = 8;
const int N_GPU_LAYERS = 99;

const int MODALITY_AR_NAR = 0;
const int MODALITY_NAR_LEN = 1;

// forward declarations
struct io_map_t;
struct llama_model;
struct llama_context;
struct encodec_context;

// model-specific parameters
struct vall_e_context_params_t {
	std::string model_path = "./data/vall_e.gguf";
	std::string encodec_path = "./data/encodec.bin";
	int32_t gpu_layers = N_GPU_LAYERS;
	int32_t n_threads = N_THREADS;
	int32_t ctx_size = CTX_SIZE;
	bool verbose = false;
};
// inference-specific arguments
struct vall_e_args_t {
	std::string text = "Hello world.";
	std::string prompt_path = "./data/prom.wav";
	std::string output_path = "./data/resp.wav";
	std::string language = "en";
	std::string task = "tts";
	int modality = MODALITY_NAR_LEN;
	int max_steps = 30;
	int max_duration = MAX_DURATION;
};
// stores everything needed for vall_e.cpp at runtime
struct vall_e_context_t {
	vall_e_context_params_t params;

	io_map_t* io_map = NULL; // pointer for reasons

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
	std::string lang = "en";

	token_t rvq_l = 0;

	std::vector<token_t> phn = {};
	vall_e_audio_codes_t prom = {};
	vall_e_audio_codes_t resp = {};
};

// encodec helpers
VALL_E_API std::vector<float> read_audio_from_disk( const std::string& path );
VALL_E_API void write_audio_to_disk( const std::vector<float>& waveform, const std::string& path );

VALL_E_API std::vector<std::vector<int32_t>> encode_audio( struct encodec_context* ectx, const std::vector<float>& waveform );
VALL_E_API std::vector<float> decode_audio( struct encodec_context* ectx, const vall_e_audio_codes_t& codes_2d );

// context management
VALL_E_API void vall_e_print_usage( char** argv, const vall_e_context_params_t& params, const vall_e_args_t& args );
VALL_E_API bool vall_e_args_parse( int argc, char** argv, vall_e_context_params_t& params, vall_e_args_t& args );
VALL_E_API vall_e_context_t* vall_e_load( const vall_e_context_params_t& params );
VALL_E_API vall_e_inputs_t vall_e_prepare_inputs( vall_e_context_t* ctx, const std::string& text, const std::string& prompt_path, const std::string& lang = "auto", const std::string& task = "tts" );
VALL_E_API vall_e_audio_codes_t vall_e_generate( vall_e_context_t* ctx, vall_e_inputs_t& inputs, int max_steps, int max_duration, int modality = MODALITY_NAR_LEN );
VALL_E_API void vall_e_free( vall_e_context_t* ctx );
