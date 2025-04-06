#define DR_WAV_IMPLEMENTATION
#include "vall_e.h"
#include "vall_e-impl.h" // stores everything that isn't necessary for exporting

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <regex>
#include <codecvt>

// this technically can be used to initialize the map directly
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

// lang map
std::unordered_map<std::string, token_t> lang_map = {
	{ "en", 0 },
	{ "ja", 1 },
	{ "de", 2 },
	{ "fr", 3 },
	{ "zh", 4 },
	{ "ko", 5 },
};
std::unordered_map<std::string, token_t> task_map = {
	{ "tts", 0 },
	{ "tts-c", 1 },
	{ "ns", 2 },
	{ "sr", 3 },
	{ "tse", 4 },
	{ "soe", 5 },
	{ "mask", 6 },
	{ "eoe", 7 },
	{ "stt", 8 },

	{ "len", 0 },
	{ "nse", 6 },
	{ "cse", 6 },
};

// u32string because encoding agony
std::unordered_map<std::u32string, token_t> vocab = {	
	{U"<unk>",0},{U"<bos>",1},{U"</eos>",2},{U"<mask>",3},{U" ",4},{U"ᵝ",4},{U"!",5},{U"\"",6},{U"(",7},{U"{",7},{U"[",7},{U")",8},{U"}",8},{U"]",8},{U",",9},{U"-",10},{U".",11},{U"1",211},{U"—",10},{U"“",6},{U"”",81},{U"ˇ",6},{U"ˉ",12},{U"ˊ",79},{U"ˋ",80},{U"_",81},{U":",13},{U";",14},{U"?",15},{U"a",16},{U"ä",16},{U"ɒ",16},{U"b",17},{U"c",18},{U"d",19},{U"e",20},{U"f",21},{U"h",22},{U"i",23},{U"ĩ",23},{U"j",24},{U"k",25},{U"l",26},{U"m",27},{U"n",28},{U"ɴ",28},{U"ɲ",28},{U"o",29},{U"̞",29},{U"p",30},{U"ɸ",30},{U"q",31},{U"r",32},{U"ɽ",32},{U"ʁ",32},{U"s",33},{U"t",34},{U"u",35},{U"ø",35},{U"œ",35},{U"y",35},{U"ɣ",35},{U"ũ",35},{U"v",36},{U"w",37},{U"ʍ",37},{U"x",38},{U"z",39},{U"¡",40},{U"«",41},{U"»",42},{U"¿",43},{U"æ",44},{U"ç",45},{U"ð",46},{U"ŋ",47},{U"ɐ",48},{U"ɑ",49},{U"ɔ",50},{U"ɕ",51},{U"ə",52},{U"ɚ",53},{U"ɛ",54},{U"ɜ",55},{U"ɟ",56},{U"ɡ",57},{U"ɪ",58},{U"ɬ",59},{U"ɯ",60},{U"ɹ",61},{U"ɾ",62},{U"ʃ",63},{U"ʈ",64},{U"ʊ",65},{U"ʋ",66},{U"ʌ",67},{U"ʑ",68},{U"ʒ",69},{U"ʔ",70},{U"ʲ",71},{U"ˈ",72},{U"ˌ",73},{U"ː",74},{U"̃",75},{U"̩",76},{U"θ",77},{U"ᵻ",78},{U"…",82},{U"ˈɛ",83},{U"iː",84},{U"aɪ",85},{U"nd",86},{U"ˈɪ",87},{U"eɪ",88},{U"ˈæ",89},{U"ðə",90},{U"oʊ",91},{U"ɑː",92},{U"ˈeɪ",93},{U"ən",94},{U"uː",95},{U"ˈʌ",96},{U"ˈaɪ",97},{U"st",98},{U"ˈɔ",99},{U"ˈoʊ",100},{U"ˈiː",101},{U"ˈɑː",102},{U"ænd",103},{U"ːɹ",104},{U"ɪŋ",105},{U"ɜː",106},{U"ɪn",107},{U"tə",108},{U"ʌv",109},{U"aʊ",110},{U"əl",111},{U"ˈuː",112},{U"tʃ",113},{U"ɪz",114},{U"ˈɜː",115},{U"ˌʌ",116},{U"æt",117},{U"dʒ",118},{U"ˈɔː",119},{U"ɪt",120},{U"ˈaʊ",121},{U"ɚɹ",122},{U"ˈɛn",123},{U"wʌ",124},{U"li",125},{U"hiː",126},{U"ˌɛ",127},{U"wɪ",128},{U"wʌz",129},{U"ðæt",130},{U"juː",131},{U"oːɹ",132},{U"ðɪ",133},{U"sˈɛ",134},{U"ˌɪ",135},{U"ˈɑːɹ",136},{U"nt",137},{U"ˈʊ",138},{U"ənt",139},{U"hɪz",140},{U"ˌɑː",141},{U"hæ",142},{U"ɔːɹ",143},{U"ˈɛɹ",144},{U"wɪð",145},{U"ᵻd",146},{U"ˈoːɹ",147},{U"pɹ",148},{U"ˈɔːl",149},{U"mˌ",150},{U"ʃən",151},{U"kt",152},{U"ˌoʊ",153},{U"ˈɔːɹ",154},{U"fɹ",155},{U"æz",156},{U"ˌʌt",157},{U"ʃiː",158},{U"ˈɛl",159},{U"ˌaʊ",160},{U"ˈʌn",161},{U"əs",162},{U"hɜː",163},{U"lˈaɪ",164},{U"ˈæn",165},{U"ˈɪɹ",166},{U"ʊd",167},{U"ɹᵻ",168},{U"ld",169},{U"bˌʌt",170},{U"ks",171},{U"nˈoʊ",172},{U"hæd",173},{U"ɾɚ",174},{U"ɛɹ",175},{U"ˈɪŋ",176},{U"ɡɹ",177},{U"nˌɑː",178},{U"ɔn",179},{U"vɚ",180},{U"maɪ",181},{U"fɔːɹ",182},{U"ðɚ",183},{U"tʊ",184},{U"ðɛɹ",185},{U"nˌɑːt",186},{U"ˈʌm",187},{U"tɹ",188},{U"sˈiː",189},{U"ʌvðə",190},{U"mˈɪ",191},{U"hˈæ",192},{U"ˌɪm",193},{U"lˈeɪ",194},{U"ɪk",195},{U"sp",196},{U"hˌɪm",197},{U"ɐn",198},{U"ðeɪ",199},{U"lˈɪ",200},{U"ɾi",201},{U"lˈɛ",202},{U"bɹ",203},{U"kɹ",204},{U"lˈæ",205},{U"ˈɪl",206},{U"jˈuː",207},{U"ʌm",208},{U"mˌiː",209},{U"bᵻ",210},{U"wˈʌn",211},{U"ˌɪn",212},{U"ˈɪn",213},{U"ˈoʊn",214},{U"sˈɛd",215},{U"biː",216},{U"ˈɛd",217},{U"ˈaɪt",218},{U"baɪ",219},{U"fɹʌm",220},{U"ɪs",221},{U"ɚz",222},{U"ðɪs",223},{U"əns",224},{U"bəl",225},{U"ɪf",226},{U"ɪnðə",227},{U"əm",228},{U"ᵻz",229},{U"ˌuː",230},{U"wˈeɪ",231},{U"ft",232},{U"wiː",233},{U"stɹ",234},{U"lˈiː",235},{U"iːz",236},{U"pt",237},{U"jʊ",238},{U"ɚd",239},{U"ˌaɪ",240},{U"kw",241},{U"ˌɔn",242},{U"ˈaɪd",243},{U"ɪm",244},{U"ˈʌst",245},{U"ˈoʊld",246},{U"ts",247},{U"ˌɪtʃ",248},{U"sˌoʊ",249},{U"dˈɪ",250},{U"ɑːɹ",251},{U"hɐ",252},{U"sˈeɪ",253},{U"ɾᵻd",254},{U"wˌɪtʃ",255},
};
// cringe list of merges to later process and fill out the map for referencing merges
std::vector<merge_entry_t> vocab_merges = {
	{U"ˈ", U"ɛ"},{U"i", U"ː"},{U"a", U"ɪ"},{U"n", U"d"},{U"ˈ", U"ɪ"},{U"e", U"ɪ"},{U"ˈ", U"æ"},{U"ð", U"ə"},{U"o", U"ʊ"},{U"ɑ", U"ː"},{U"ˈ", U"eɪ"},{U"ə", U"n"},{U"u", U"ː"},{U"ˈ", U"ʌ"},{U"ˈ", U"aɪ"},{U"s", U"t"},{U"ˈ", U"ɔ"},{U"ˈ", U"oʊ"},{U"ˈ", U"iː"},{U"ˈ", U"ɑː"},{U"æ", U"nd"},{U"ː", U"ɹ"},{U"ɪ", U"ŋ"},{U"ɜ", U"ː"},{U"ɪ", U"n"},{U"t", U"ə"},{U"ʌ", U"v"},{U"a", U"ʊ"},{U"ə", U"l"},{U"ˈ", U"uː"},{U"t", U"ʃ"},{U"ɪ", U"z"},{U"ˈ", U"ɜː"},{U"ˌ", U"ʌ"},{U"æ", U"t"},{U"d", U"ʒ"},{U"ˈɔ", U"ː"},{U"ɪ", U"t"},{U"ˈ", U"aʊ"},{U"ɚ", U"ɹ"},{U"ˈɛ", U"n"},{U"w", U"ʌ"},{U"l", U"i"},{U"h", U"iː"},{U"ˌ", U"ɛ"},{U"w", U"ɪ"},{U"wʌ", U"z"},{U"ð", U"æt"},{U"j", U"uː"},{U"o", U"ːɹ"},{U"ð", U"ɪ"},{U"s", U"ˈɛ"},{U"ˌ", U"ɪ"},{U"ˈɑː", U"ɹ"},{U"n", U"t"},{U"ˈ", U"ʊ"},{U"ən", U"t"},{U"h", U"ɪz"},{U"ˌ", U"ɑː"},{U"h", U"æ"},{U"ɔ", U"ːɹ"},{U"ˈɛ", U"ɹ"},{U"wɪ", U"ð"},{U"ᵻ", U"d"},{U"ˈ", U"oːɹ"},{U"p", U"ɹ"},{U"ˈɔː", U"l"},{U"m", U"ˌ"},{U"ʃ", U"ən"},{U"k", U"t"},{U"ˌ", U"oʊ"},{U"ˈɔ", U"ːɹ"},{U"f", U"ɹ"},{U"æ", U"z"},{U"ˌʌ", U"t"},{U"ʃ", U"iː"},{U"ˈɛ", U"l"},{U"ˌ", U"aʊ"},{U"ˈʌ", U"n"},{U"ə", U"s"},{U"h", U"ɜː"},{U"l", U"ˈaɪ"},{U"ˈæ", U"n"},{U"ˈɪ", U"ɹ"},{U"ʊ", U"d"},{U"ɹ", U"ᵻ"},{U"l", U"d"},{U"b", U"ˌʌt"},{U"k", U"s"},{U"n", U"ˈoʊ"},{U"hæ", U"d"},{U"ɾ", U"ɚ"},{U"ɛ", U"ɹ"},{U"ˈɪ", U"ŋ"},{U"ɡ", U"ɹ"},{U"n", U"ˌɑː"},{U"ɔ", U"n"},{U"v", U"ɚ"},{U"m", U"aɪ"},{U"f", U"ɔːɹ"},{U"ð", U"ɚ"},{U"t", U"ʊ"},{U"ð", U"ɛɹ"},{U"nˌɑː", U"t"},{U"ˈʌ", U"m"},{U"t", U"ɹ"},{U"s", U"ˈiː"},{U"ʌv", U"ðə"},{U"m", U"ˈɪ"},{U"h", U"ˈæ"},{U"ˌɪ", U"m"},{U"l", U"ˈeɪ"},{U"ɪ", U"k"},{U"s", U"p"},{U"h", U"ˌɪm"},{U"ɐ", U"n"},{U"ð", U"eɪ"},{U"l", U"ˈɪ"},{U"ɾ", U"i"},{U"l", U"ˈɛ"},{U"b", U"ɹ"},{U"k", U"ɹ"},{U"l", U"ˈæ"},{U"ˈɪ", U"l"},{U"j", U"ˈuː"},{U"ʌ", U"m"},{U"mˌ", U"iː"},{U"b", U"ᵻ"},{U"w", U"ˈʌn"},{U"ˌ", U"ɪn"},{U"ˈɪ", U"n"},{U"ˈoʊ", U"n"},{U"sˈɛ", U"d"},{U"b", U"iː"},{U"ˈɛ", U"d"},{U"ˈaɪ", U"t"},{U"b", U"aɪ"},{U"fɹ", U"ʌm"},{U"ɪ", U"s"},{U"ɚ", U"z"},{U"ðɪ", U"s"},{U"ən", U"s"},{U"b", U"əl"},{U"ɪ", U"f"},{U"ɪn", U"ðə"},{U"ə", U"m"},{U"ᵻ", U"z"},{U"ˌ", U"uː"},{U"w", U"ˈeɪ"},{U"f", U"t"},{U"w", U"iː"},{U"st", U"ɹ"},{U"l", U"ˈiː"},{U"iː", U"z"},{U"p", U"t"},{U"j", U"ʊ"},{U"ɚ", U"d"},{U"ˌ", U"aɪ"},{U"k", U"w"},{U"ˌ", U"ɔn"},{U"ˈaɪ", U"d"},{U"ɪ", U"m"},{U"ˈʌ", U"st"},{U"ˈoʊ", U"ld"},{U"t", U"s"},{U"ˌɪ", U"tʃ"},{U"s", U"ˌoʊ"},{U"d", U"ˈɪ"},{U"ɑː", U"ɹ"},{U"h", U"ɐ"},{U"s", U"ˈeɪ"},{U"ɾ", U"ᵻd"},{U"w", U"ˌɪtʃ"},
};
// merge map to reference when tokenizing text
std::unordered_map<std::string, merge_entry_t> vocab_merge_map = {};

std::vector<float> read_2d_tensor( struct ggml_tensor* tensor ) {
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

ggml_tensor* view_2d_tensor( struct ggml_context* ctx, struct ggml_tensor* tensor, int32_t start, int32_t end, int32_t dim ) {
	// to-do: implement other dim
	if ( start < 0 ) start = tensor->ne[1] + start;
	if ( end < 0 ) end = tensor->ne[1] + end;

	ggml_tensor* res = ggml_view_2d( ctx, tensor, tensor->ne[0], end - start, tensor->nb[1], tensor->nb[1] * start );

	return res;
}

void print_tokens( const std::vector<token_t>& tokens, const std::string& prefix ) {
	printf("%s[", prefix.c_str());
	for ( auto i = 0; i < tokens.size(); ++i ) {
		printf("%i%s", tokens[i], i + 1 < tokens.size() ? ", " : "");
	}
	printf("]\n");
}
void print_floats( const std::vector<float>& v, const std::string& prefix ) {
	printf("%s[", prefix.c_str());
	for ( auto i = 0; i < v.size(); ++i ) {
		printf("%f%s", v[i], i + 1 < v.size() ? ", " : "");
	}
	printf("]\n");
}


float calculate_std(const float* data, size_t n) {
	float mean = 0.0f;
	for (size_t i = 0; i < n; i++) mean += data[i];
	mean /= n;

	float variance = 0.0f;
	for (size_t i = 0; i < n; i++) {
		float diff = data[i] - mean;
		variance += diff * diff;
	}
	variance /= n;

	return sqrt(variance);
}

const io_t& vall_e_inputs_map_get( io_map_t& io_map, const std::string& name ) {
	return io_map.io[name];
}
const float* vall_e_inputs_map_get_embeddings_p( io_map_t& io_map, const std::string& name ) {
	return io_map.io[name].embds.data();	
}

int32_t vall_e_inputs_map_get_classifier_idx( io_map_t& io_map, const std::string& name ) {
	return io_map.io[name].head_idx;
}

void vall_e_inputs_map_init( io_map_t& io_map, llama_model* model ) {
	auto vocab = llama_model_get_vocab( model );
	auto n_embd = llama_model_n_embd( model );
	auto n_vocab = llama_vocab_n_tokens( vocab );
	
	io_map.n_embd = n_embd;
	io_map.n_vocab = n_vocab;

	size_t ctx_size = 24 * 2 * ggml_tensor_overhead(); // 24 embeddings + 24 output heads (generous) (should only really need to do this for output heads since we manually handle embeddings)
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
std::vector<std::vector<float>> map_embeddings( const std::vector<token_t>& tokens, int n_embd, const float* embds ) {
	std::vector<std::vector<float>> embedded( tokens.size() );
	for ( auto i = 0; i < tokens.size(); ++i ) {
		embedded[i].insert( embedded[i].end(), embds + (tokens[i] * n_embd), embds + ((tokens[i]+1) * n_embd) );
	}
	return embedded;
}

// handles adding either a token OR the embedding of that token into the batch
// this really, really helps avoid needing to abuse the tokenizer
void batch_add( llama_batch& batch, token_t id, int n_embd, const float* embds, llama_pos pos, bool output, const std::vector<llama_seq_id> & seq_ids ) {
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
std::vector<float> read_audio_from_disk( const std::string& path ) {
	std::vector<float> res;

	uint32_t channels;
	uint32_t sample_rate;
	drwav_uint64 total_frame_count;

	float * raw_audio = drwav_open_file_and_read_pcm_frames_f32(path.c_str(), &channels, &sample_rate, &total_frame_count, NULL);

	if (raw_audio == NULL) {
		fprintf(stderr, "%s: could not read wav file\n", __func__);
		return res;
	}

	if (sample_rate != 24000) {
		fprintf(stderr, "%s: wav file is wrong sample rate\n", __func__);
		return res;
	}

	fprintf(stderr, "\n%s: Number of frames read = %lld.\n", __func__, total_frame_count);

	res.resize(total_frame_count);
	memcpy(res.data(), raw_audio, total_frame_count * sizeof(float));

	drwav_free(raw_audio, NULL);

	return res;
}
// writes a waveform to disk
void write_audio_to_disk( const std::vector<float>& wavform, const std::string& path ) {
	drwav_data_format format;
	format.bitsPerSample = 32;
	format.sampleRate = 24000;
	format.container = drwav_container_riff;
	format.channels = 1;
	format.format = DR_WAVE_FORMAT_IEEE_FLOAT;

	drwav wav;
	drwav_init_file_write(&wav, path.c_str(), &format, NULL);
	drwav_uint64 frames = drwav_write_pcm_frames(&wav, wavform.size(), wavform.data());
	drwav_uninit(&wav);

	fprintf(stderr, "%s: Number of frames written = %lld.\n", __func__, frames);
}
// reads a waveform from disk then encodes it
std::vector<std::vector<int32_t>> encode_audio( struct encodec_context* ectx, const std::vector<float>& wavform ) {
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
std::vector<float> decode_audio( struct encodec_context* ectx, const std::vector<std::vector<int32_t>>& codes ) {
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
std::vector<std::vector<float>> sum_embeddings( const std::vector<std::vector<token_t>>& inputs, int n_embd, int rvq_l, const float** embds, int mode ) {
	auto n_tokens = inputs[0].size();

	std::vector<std::vector<float>> res( n_tokens, std::vector<float>( n_embd, 0.0 ) );

	// iterate through rvq levels (only up to inclusive the target rvq level)
	for ( auto l = 0; l < inputs.size() && l <= rvq_l; ++l ) {
		int offset = 0;
		// handles the cringe logic I have
		if ( mode == EMBEDDING_MODE_RESP_AR_NAR ) {
			offset = inputs.size() == 1 ? 0 : 1;
		} else if ( mode == EMBEDDING_MODE_RESP_NAR_LEN ) {
			offset = inputs.size() == 1 ? 8 : 1;
		}
		// embed the current level's tokens
		auto embedded = map_embeddings( inputs[l], n_embd, embds[l + offset] );

		for ( auto idx = 0; idx < n_tokens; ++idx ) {
			for ( auto embd_idx = 0; embd_idx < n_embd; ++embd_idx ) {
				res[idx][embd_idx] += embedded[idx][embd_idx];
			}
		}
	}
	return res;
}

std::vector<float> soft_max( int n_logits, const float* logits ) {
	std::vector<float> res(n_logits, 0.0f);
	float denom = 0.0f;

	float max_logit = logits[0];
	for (int i = 1; i < n_logits; ++i) {
		max_logit = std::max(max_logit, logits[i]);
	}

	for (int i = 0; i < n_logits; ++i) {
		res[i] = std::exp(logits[i] - max_logit);
		denom += res[i];
	}

	float inv_denom = 1.0f / denom;
	for (int i = 0; i < n_logits; ++i) {
		res[i] *= inv_denom;
	}

	return res;
}

void fill_batch( llama_batch& batch, vall_e_inputs_t& inputs, io_map_t& io_map, int mode ) {
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

	token_t lang_token = lang_map[inputs.lang];
	token_t task_token = task_map[inputs.task];

	// insert text tokens
	for ( auto& id : inputs.phn ) batch_add( batch, id, n_embd, text_embds, pos++, false );
	batch_add( batch, 0, n_embd, sep_embds, pos++, false );
	pos = 0;
	// insert lang token
	batch_add( batch, lang_token, n_embd, lang_embds, pos++, false );
	batch_add( batch, 0, n_embd, sep_embds, pos++, false );
	pos = 0;
	// insert rvq level token
	batch_add( batch, inputs.rvq_l, n_embd, rvq_l_embds, pos++, false );
	batch_add( batch, 0, n_embd, sep_embds, pos++, false );
	pos = 0;
	// input task token if needed
	if ( task_token > 0 ) {
		batch_add( batch, task_token, n_embd, task_embds, pos++, false );
	}
	// insert prom tokens
	auto summed_proms_embds = sum_embeddings( inputs.prom, n_embd, inputs.rvq_l, prom_embds );
	for ( auto i = 0; i < summed_proms_embds.size(); ++i ) {
		batch_add( batch, -1, n_embd, summed_proms_embds[i].data(), pos++, false );
	}
	batch_add( batch, 0, n_embd, sep_embds, pos++, mode == INFERENCE_MODE_AR ); // set as the last logit if AR
	pos = 0;

	// inputs starting len token
	if ( inputs.task == "len" ) {
		batch_add( batch, 0, n_embd, len_embds, pos++, true );
		pos = 0;
	}

	// insert resp tokens
	if ( !inputs.resp.empty() ) {
		auto summed_resps_embds = sum_embeddings( inputs.resp, n_embd, inputs.rvq_l, resp_embds, mode == INFERENCE_MODE_AR ? EMBEDDING_MODE_RESP_AR_NAR : EMBEDDING_MODE_RESP_NAR_LEN );
		for ( auto i = 0; i < summed_resps_embds.size(); ++i ) {
			batch_add( batch, -1, n_embd, &summed_resps_embds[i][0], pos++, true );
		}
		pos = 0;
	}
}

// generation code, should handle all modalities easily
std::vector<token_t> generate( vall_e_context_t* ctx, vall_e_inputs_t& inputs, int max_tokens, int mode, bool verbose ) {
	bool causal = true; // sample autoregressively or not
	int n_outputs = 0; // number of output tokens to expect

	// create batch	(targetting embeddings instead of tokens)
	llama_batch batch = llama_batch_init( ctx->params.ctx_size, ctx->io_map->n_embd, ctx->params.ctx_size );
	fill_batch( batch, inputs, *ctx->io_map, mode );

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
		embd_name = k_embds[inputs.rvq_l];
	// duration inferencing mode
	} else if ( mode == INFERENCE_MODE_LEN ) {
		embd_name = "len";
	// NAR-len (demasking) inferencing mode
	} else if ( mode == INFERENCE_MODE_NAR_DEMASK ) {
		embd_name = "resps|NAR:0:0";
	}

	auto& io = vall_e_inputs_map_get(*ctx->io_map, embd_name);
	const float* embds = io.embds.data();

	int32_t n_embd = io.n_embd;
	int32_t n_vocab = io.n_vocab;
	token_t stop_token = io.end - io.start - 1;

	if ( verbose ) printf("Generating in %s (%i) mode (%i:%i) (%i)\n", embd_name.c_str(), io.head_idx, io.start, io.end, stop_token);

	// update model's output heads / causal mode
	llama_set_output_head( ctx->llama.model, io.head );
	// to-do: figure this out......
	{
		llama_set_causal_attn( ctx->llama.ctx, causal ); // to-do: fix GGML_ASSERT(mask->ne[0] == a->ne[0])
		*const_cast<bool*>(&ctx->llama.model->hparams.causal_attn) = true; // force set this
	}

	std::vector<token_t> output_tokens;
	const auto t_main_start = ggml_time_us();

	// if INFERENCE_MODE_AR || INFERENCE_MODE_LEN
	if ( causal ) {
		auto sparams = llama_sampler_chain_default_params();
		sparams.no_perf = false;
		llama_sampler * smpl = llama_sampler_chain_init(sparams);

		if ( mode == INFERENCE_MODE_LEN ) {
			llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
		} else {
			llama_sampler_chain_add(smpl, llama_sampler_init_top_k(0));
			llama_sampler_chain_add(smpl, llama_sampler_init_top_p(1.0, 1));
			llama_sampler_chain_add(smpl, llama_sampler_init_temp (1.0));
			llama_sampler_chain_add(smpl, llama_sampler_init_dist (LLAMA_DEFAULT_SEED));
		}

		output_tokens.reserve(max_tokens);
		while ( output_tokens.size() < max_tokens ) {
			if ( llama_decode(ctx->llama.ctx, batch) ) {
				fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
				return output_tokens;
			}
			llama_kv_self_clear(ctx->llama.ctx); // necessary for many reasons

			// sample token
			auto t = llama_sampler_sample(smpl, ctx->llama.ctx, -1);

			// is stop token
			if ( t == stop_token ) {
				break;
			}

			// store token
			output_tokens.emplace_back(t);
			// update batch with token
			batch_add( batch, t, ctx->io_map->n_embd, embds, output_tokens.size(), true );
			
			if ( verbose ) print_tokens( output_tokens );
		}

		llama_sampler_free(smpl);
	} else if ( mode == INFERENCE_MODE_NAR_DEMASK ) {
		// to-do: assert n_outputs == inputs.resp[rvq_l-1].size()
		const token_t MASK_TOKEN = 1024; // token value for masking 
		const float PI = 3.141592653589793f;
		
		// to-do: derive from sampling arguments
		int32_t steps = max_tokens;
		int32_t seq_len = n_outputs;
		int32_t top_k = 0;
		float top_p = 1.0;
		float temperature = 1.0f;
		float cfg_strength = 3.0f;
		float start_noise = 0.0f;
		float end_noise = 1.0f;
		bool annealed_sampling = true;
		bool remasking = true;
		float cfg_rescale = 0.75f;
		bool entropy_scoring = true;

		// fill with masked tokens
		output_tokens.clear();
		output_tokens.resize(n_outputs, MASK_TOKEN);

		// for CFG
		vall_e_inputs_t null_input{};
		null_input.phn = {1, 2}; // <bos></eos>
		null_input.resp.resize(1);

		llama_batch null_batch = llama_batch_init( ctx->params.ctx_size, ctx->io_map->n_embd, ctx->params.ctx_size );
		
		// token scores to reference for masking
		std::vector<float> scores(n_outputs, entropy_scoring ? 0.0 : 1.0);

		// do one step on many tokens
		for ( auto step = 0; step < steps; ++step ) {
			float t_norm = static_cast<float>(step) / static_cast<float>(steps - 1);
			float timestep = start_noise + (end_noise - start_noise) * t_norm;
			//float timestep = start_noise + (end_noise - start_noise) * ((float)step / steps);
			
			float annealing = 1.0f - timestep;

			float sampling_temperature = annealed_sampling ? temperature * annealing : temperature;
			float sampling_cfg_strength = annealed_sampling ? timestep * cfg_strength : cfg_strength;

			float noise_p = cos( timestep * PI * 0.5f );
			float remask_p = remasking ? 1.0f / (steps * 2.0f) : 0.0f;
			
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
			if ( entropy_scoring) {
				std::reverse(sorted_scores.begin(), sorted_scores.end());
			}
			
			// and top-k pick the worst scores
			for ( auto i = 0; i < n_masked_tokens; ++i ) {
				auto idx = sorted_scores[i].idx;

				output_tokens[idx] = MASK_TOKEN;
				is_masked[idx] = true;
			}

			if ( verbose ) print_tokens( output_tokens, std::string("[")+std::to_string(step)+"/"+std::to_string(steps)+"] Masked tokens: " );

			// update batch
			// to-do: only update the embeddings instead
			batch.n_tokens = 0;
			inputs.resp[0] = output_tokens;
			fill_batch( batch, inputs, *ctx->io_map, mode );
			// update null batch
			null_batch.n_tokens = 0;
			null_input.resp[0] = output_tokens;
			fill_batch( null_batch, inputs, *ctx->io_map, mode );

			// cfg decode
			if ( llama_decode(ctx->llama.ctx, null_batch) ) {
				fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
				return output_tokens;
			}
			llama_kv_self_clear(ctx->llama.ctx); // necessary for many reasons
			// copy null probabilities
			std::vector<float> null_logits(n_outputs * n_vocab, 0.0f);
			memcpy( null_logits.data(), llama_get_logits( ctx->llama.ctx ), sizeof(float) * n_vocab * n_outputs );

			// decode
			if ( llama_decode(ctx->llama.ctx, batch) ) {
				fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
				return output_tokens;
			}
			llama_kv_self_clear(ctx->llama.ctx); // necessary for many reasons
			
			auto sparams = llama_sampler_chain_default_params();
			sparams.no_perf = false;
			llama_sampler * smpl = llama_sampler_chain_init(sparams);

			if ( sampling_temperature == 0 ) {
				llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
			} else {
				llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
				llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
				llama_sampler_chain_add(smpl, llama_sampler_init_temp (sampling_temperature));
				llama_sampler_chain_add(smpl, llama_sampler_init_dist (LLAMA_DEFAULT_SEED));
			}

			auto* logits = llama_get_logits( ctx->llama.ctx );
			for ( auto idx = 0; idx < n_outputs; ++idx ) {
				// skip if not masked
				if ( !is_masked[idx] ) {
					scores[idx] = entropy_scoring ? 0.0 : 1.0;
					continue;
				}

				auto* logit = &logits[idx * n_vocab];
				auto* null_logit = &null_logits[idx * n_vocab];

				// perform softmax before modifying logits
				std::vector<float> softmaxed = soft_max( n_vocab, logit );
				int32_t t_u = std::distance( softmaxed.begin(), std::max_element(softmaxed.begin(), softmaxed.end()) );

				std::vector<float> summed(n_vocab);
				for (int i = 0; i < n_vocab; i++) {
					summed[i] = null_logit[i] + (logit[i] - null_logit[i]) * sampling_cfg_strength;
				}

				if (cfg_rescale > 0) {
					float pos_std = calculate_std(logit, n_vocab);
					float summed_std = calculate_std(summed.data(), n_vocab);
					float factor = cfg_rescale * (pos_std / summed_std) + (1 - cfg_rescale);

					for (int i = 0; i < n_vocab; i++) {
						logit[i] = summed[i] * factor;
					}
				} else {
					memcpy(logit, summed.data(), n_vocab * sizeof(float));
				}

				// sample ith token
				auto t = llama_sampler_sample(smpl, ctx->llama.ctx, batch.n_tokens - n_outputs + idx );
				// store token if it was masked
				output_tokens[idx] = t;
				// update score if it was masked
				if ( entropy_scoring ) {
					float entropy = 0.f;
					for (int v = 0; v < n_vocab; ++v ) {
						float p = softmaxed[v];
						if (p > 0) entropy -= p * std::log(p + 1e-9);
					}
					scores[idx] = entropy / std::log(n_vocab); // normalize [0–1]
				} else {
					scores[idx] = softmaxed[t_u]; // invert so we pick the worst tokens later
				}
			}

			llama_sampler_free(smpl);

			if ( verbose ) print_tokens( output_tokens, std::string("[")+std::to_string(step)+"/"+std::to_string(steps)+"]: " );
		}
	} else if ( mode == INFERENCE_MODE_NAR ) {
		// to-do: assert n_outputs == inputs.resp[rvq_l-1].size()
		output_tokens.clear();
		output_tokens.resize(n_outputs);
		// do one step on many tokens
		if ( llama_decode(ctx->llama.ctx, batch) ) {
			fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
			return output_tokens;
		}
		llama_kv_self_clear(ctx->llama.ctx); // necessary for many reasons

		auto sparams = llama_sampler_chain_default_params();
		sparams.no_perf = false;
		llama_sampler * smpl = llama_sampler_chain_init(sparams);

		llama_sampler_chain_add(smpl, llama_sampler_init_top_k(20));
		llama_sampler_chain_add(smpl, llama_sampler_init_top_p(1.0, 1));
		llama_sampler_chain_add(smpl, llama_sampler_init_temp (1.0));
		llama_sampler_chain_add(smpl, llama_sampler_init_dist (LLAMA_DEFAULT_SEED));

		for ( auto idx = 0; idx < n_outputs; ++idx ) {
			// sample ith token
			auto t = llama_sampler_sample(smpl, ctx->llama.ctx, batch.n_tokens - n_outputs + idx);

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
		llama_perf_context_print(ctx->llama.ctx);
		fprintf(stderr, "\n");
	}
	
	llama_batch_free(batch);

	return output_tokens;
}

std::vector<token_t> phonemize( vall_e_context_t* ctx, const std::string& text, const std::string& language ) {	
	std::vector<token_t> tokens;

	// phonemize text
	std::string espeak_language = "en";
	if ( language == "en" ) espeak_language = "en-us";
	else if ( language == "fr" ) espeak_language = "fr-fr";
	else if ( language == "zh" ) espeak_language = "cmn-latn-pinyin";
	espeak_SetVoiceByName(espeak_language.c_str());

	const char* text_c_str = text.c_str();
	const char* phonemes = espeak_TextToPhonemes((const void**) &text_c_str, espeakCHARS_UTF8, espeakPHONEMES_IPA);

	std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv_utf8_utf32;
	std::u32string unicode_phonemes = conv_utf8_utf32.from_bytes(phonemes);

	// manual tokenization because llama tokenizer isn't cooperating
	// to-do: handle merges
	tokens.emplace_back(1);
	for (auto& phone : unicode_phonemes ) {
		std::u32string phone_str;
		phone_str += phone;
		// place <unk> first
		auto& token = tokens.emplace_back(0);
		// update if found
		if ( vocab.count( phone_str ) > 0 ) {
			token = vocab[phone_str];
		}
	}
	
	// handle merges (skip <bos>)
	for ( auto i = 1; i < tokens.size() - 1; ++i ) {
		auto& cur = tokens[i];
		auto& next = tokens[i+1];
		std::string key = std::to_string(cur) + ":" + std::to_string(next);
		// not a merge
		if ( !vocab_merge_map.count(key) )
			continue;

		// get merge entry
		auto& merge = vocab_merge_map[key];
		// update with merged token
		cur = merge.resolved_token;
		// erase at next token
		tokens.erase(tokens.begin() + i + 1);
		// back iterate to check for more merges at next iteration
		--i;
	}
	tokens.emplace_back(2);

	if ( ctx->params.verbose ) print_tokens( tokens, "Phonemes: " );

	/*
	// to-do: fix terminate called after throwing an instance of 'std::out_of_range'
	// deduce token count
	const int n_tokens = -llama_tokenize(ctx->llama.model, phonemes.c_str(), phonemes.size(), NULL, 0, true, true);
	tokens.resize(n_tokens);
	// tokenize
	if ( llama_tokenize(ctx->llama.model, phonemes.c_str(), phonemes.size(), tokens.data(), tokens.size(), true, true) < 0 ) {
		fprintf(stderr, "%s: error: failed to tokenize: %s\n", __func__, phonemes.c_str());
		return tokens;
	}
	*/
	return tokens;
}

void vall_e_print_usage( char** argv, vall_e_context_params_t& params, vall_e_args_t& args ) {
	fprintf(stderr, "usage: %s [options]\n", argv[0]);
	fprintf(stderr, "\n");
	fprintf(stderr, "options:\n");
	fprintf(stderr, "  -h, --help                            Show this help message and exit\n");
	fprintf(stderr, "  -t N, --threads N\n");
	fprintf(stderr, "                                        Number of threads to use during computation (default: %d)\n", params.n_threads);
	fprintf(stderr, "  -ngl N, --n-gpu-layers N\n");
	fprintf(stderr, "                                        Number of layers to offload to the GPU (default: %d)\n", params.gpu_layers);
	fprintf(stderr, "  -ctx N, --context-size N\n");
	fprintf(stderr, "                                        Max context size (default: %d)\n", params.ctx_size);
	fprintf(stderr, "  -v, --verbose\n");
	fprintf(stderr, "                                        Verbose output (default: %d)\n", params.verbose);
	fprintf(stderr, "  -m FNAME, --model FNAME\n");
	fprintf(stderr, "                                        VALL-E model path (default: %s)\n", params.model_path.c_str());
	fprintf(stderr, "  -em FNAME, --encodec-model FNAME\n");
	fprintf(stderr, "                                        Encodec model path (default: %s)\n", params.encodec_path.c_str());
	fprintf(stderr, "  -t TEXT, --text TEXT\n");
	fprintf(stderr, "                                        Input text prompt (default: %s)\n", args.text.c_str());
	fprintf(stderr, "  -l TEXT, --language TEXT\n");
	fprintf(stderr, "                                        Language for input text / output response (default: %s)\n", args.language.c_str());
	fprintf(stderr, "  -ts TASK, --task TASK\n");
	fprintf(stderr, "                                        Inferencing task (default: %s, accepts ['tts', 'stt', 'ns', 'sr'])\n", args.task.c_str());
	fprintf(stderr, "  -mode MODE, --modality MODE\n");
	fprintf(stderr, "                                        Modality for inferencing (default: %s, accepts ['ar+nar', 'nar-len'])\n", args.modality == MODALITY_NAR_LEN ? "nar-len" : "ar+nar");
	fprintf(stderr, "  -ms N, --max-steps N\n");
	fprintf(stderr, "                                        Max steps for `nar-len` (default: %i)\n", args.max_steps);
	fprintf(stderr, "  -md N, --max-duration N\n");
	fprintf(stderr, "                                        Max duration of the audio (default: %i)\n", args.max_duration);
	fprintf(stderr, "  -i FNAME, --input FNAME\n");
	fprintf(stderr, "                                        Input prompt wav (default: %s)\n", args.prompt_path.c_str());
	fprintf(stderr, "  -o FNAME, --output FNAME\n");
	fprintf(stderr, "                                        Output audio wav (default: %s)\n", args.output_path.c_str());
	fprintf(stderr, "\n");
}
bool vall_e_args_parse( int argc, char** argv, vall_e_context_params_t& params, vall_e_args_t& args ) {
	for ( int i = 1; i < argc; i++ ) {
		std::string arg = argv[i];

		if (arg == "-t" || arg == "--threads") {
			params.n_threads = std::stoi(argv[++i]);
		} else if (arg == "-ngl" || arg == "--n-gpu-layers") {
			params.gpu_layers = std::stoi(argv[++i]);
		} else if (arg == "-ctx" || arg == "--context-size") {
			params.ctx_size = std::stoi(argv[++i]);
		} else if (arg == "-v" || arg == "--verbose") {
			params.verbose = true;
		} else if (arg == "-m" || arg == "--model") {
			params.model_path = argv[++i];
		} else if (arg == "-em" || arg == "--encodec-model") {
			params.encodec_path = argv[++i];
		} else if (arg == "-t" || arg == "--text") {
			args.text = argv[++i];
		} else if (arg == "-l" || arg == "--language") {
			args.language = argv[++i];
		} else if (arg == "-ts" || arg == "--task") {
			args.task = argv[++i];
		} else if (arg == "-mode" || arg == "--modality") {
			args.modality = std::string(argv[++i]) == "ar+nar" ? MODALITY_AR_NAR : MODALITY_NAR_LEN;
		} else if (arg == "-ms" || arg == "--max-steps") {
			args.max_steps = std::stoi(argv[++i]);
		} else if (arg == "-md" || arg == "--max-duration") {
			args.max_duration = std::stoi(argv[++i]) * ENCODEC_FRAMES_PER_SECOND;
		} else if (arg == "-i" || arg == "--input") {
			args.prompt_path = argv[++i];
		} else if (arg == "-o" || arg == "--output") {
			args.output_path = argv[++i];
		} else if (arg == "-h" || arg == "--help") {
			vall_e_print_usage(argv, params, args);
			exit(0);
			return false;
		} else {
			fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
			vall_e_print_usage(argv, params, args);
			exit(0);
			return false;
		}
	}
	return true;
}

vall_e_context_t* vall_e_load( const vall_e_context_params_t& params ) {
	vall_e_context_t* ctx = new vall_e_context_t();
	ctx->io_map = new io_map_t();
	ctx->params = params;

	// setup ggml
	ggml_backend_load_all();

	// setup llama.cpp
	llama_model_params model_params = llama_model_default_params();
	model_params.n_gpu_layers = params.gpu_layers;

	ctx->llama.model = llama_model_load_from_file(params.model_path.c_str(), model_params);
	if ( !ctx->llama.model ) {
		fprintf(stderr , "%s: error: unable to load model\n" , __func__);
		return ctx;
	}

	// initialize the context
	llama_context_params ctx_params = llama_context_default_params();
	ctx_params.n_ctx = params.ctx_size;
	ctx_params.n_batch = params.ctx_size;
	ctx_params.n_ubatch = params.ctx_size;
	ctx_params.n_threads = params.n_threads;
	ctx_params.n_threads_batch = params.n_threads;
	ctx_params.no_perf = false;
	ctx_params.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL; 

	ctx->llama.ctx = llama_init_from_model(ctx->llama.model, ctx_params);
	if ( !ctx->llama.ctx ) {
		fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
		return ctx;
	}
	
	// setup encodec.cpp
	ctx->encodec.ctx = encodec_load_model(params.encodec_path.c_str(), 0, params.gpu_layers);
	if ( !ctx->encodec.ctx ) {
		fprintf(stderr, "%s: error during loading model\n", __func__);
		return ctx;
	}
	encodec_set_target_bandwidth(ctx->encodec.ctx, 6);
	encodec_set_sample_rate(ctx->encodec.ctx, 24000);

	// setup espeak
	espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, NULL, 0);

	// setup vall_e.cpp
	vall_e_inputs_map_init( *ctx->io_map, ctx->llama.model );

	// setup vocab things
	for ( auto& entry : vocab_merges ) {
		entry.resolved = entry.pre+entry.post;

		entry.pre_token = vocab[entry.pre];
		entry.post_token = vocab[entry.post];
		entry.resolved_token = vocab[entry.resolved];

		std::string key = std::to_string(entry.pre_token) + ":" + std::to_string(entry.post_token);	
		vocab_merge_map[key] = entry;
	}

	return ctx;
}
vall_e_inputs_t  vall_e_prepare_inputs( vall_e_context_t* ctx, const std::string& text, const std::string& prompt_path, const std::string& language, const std::string& task ) {
	// to-do: set members in initializer rather than in post
	vall_e_inputs_t inputs;
	
	inputs.task = task;
	inputs.rvq_l = 0;
	inputs.phn = phonemize( ctx, text, language );
	inputs.prom = encode_audio( ctx->encodec.ctx, read_audio_from_disk( prompt_path ) );
	inputs.lang = language;

	return inputs;
}
// to-do: provide sampling params
vall_e_audio_codes_t vall_e_generate( vall_e_context_t* ctx, vall_e_inputs_t& inputs, int max_steps, int max_duration, int modality ) {
	// NAR-len demasking
	std::vector<token_t> output_tokens;
	if ( modality == MODALITY_NAR_LEN ) {
		// inference len
		int len = 0;
		if ( !len ) {
			auto task = inputs.task;
			inputs.task = "len";
			output_tokens = generate( ctx, inputs, 5, INFERENCE_MODE_LEN, ctx->params.verbose );
			{
				// to-do: one liner this
				int digit = 1;
				for (auto it = output_tokens.rbegin(); it < output_tokens.rend(); ++it) {
					len += (*it) * digit;
					digit *= 10;
				}
			}
			// cap duration
			if ( len <= 0 || len > max_duration ) len = max_duration;
			inputs.task = task;
		}
		// fill with mask tokens
		inputs.resp.resize(1);
		for ( auto i = 0; i < len; ++i ) {
			inputs.resp[0].emplace_back( 1024 ); // fill with masked tokens
		}

		// inference NAR-len 0
		for ( auto l = 0; l < 8; ++l ) {
			inputs.rvq_l = l;
			output_tokens = generate( ctx, inputs, max_steps, l == 0 ? INFERENCE_MODE_NAR_DEMASK  : INFERENCE_MODE_NAR, ctx->params.verbose );
			if ( l == 0 ) inputs.resp.clear();
			inputs.resp.emplace_back( output_tokens );
		}
	// AR+NAR
	} else if ( modality == MODALITY_AR_NAR ){
		for ( auto l = 0; l < 8; ++l ) {
			inputs.rvq_l = l;
			output_tokens = generate( ctx, inputs, l == 0 ? max_duration : 1, l == 0 ? INFERENCE_MODE_AR  : INFERENCE_MODE_NAR, ctx->params.verbose );
			inputs.resp.emplace_back( output_tokens );
		}
	}

	return inputs.resp;
}
void vall_e_free( vall_e_context_t* ctx ) {
	espeak_Terminate();
	encodec_free(ctx->encodec.ctx);
	llama_free(ctx->llama.ctx);
	llama_model_free(ctx->llama.model);
	ggml_free(ctx->io_map->ctx);
	delete ctx->io_map;
	delete ctx;
}

int main( int argc, char** argv ) {
	vall_e_context_params_t params;
	vall_e_args_t args;

	if ( !vall_e_args_parse( argc, argv, params, args ) ) {
		fprintf(stderr, "%s: failed to parse arguments\n", __func__);
		return 1;
	}

	vall_e_context_t* ctx = vall_e_load( params );
	if ( !ctx || !ctx->llama.model || !ctx->llama.ctx || !ctx->encodec.ctx  ) {
		fprintf(stderr, "%s: failed to initialize vall_e.cpp\n", __func__);
		return 1;
	}

	auto inputs = vall_e_prepare_inputs( ctx, args.text, args.prompt_path, args.language );
	auto output_audio_codes = vall_e_generate( ctx, inputs, args.max_steps, args.max_duration, args.modality );
	write_audio_to_disk( decode_audio( ctx->encodec.ctx, output_audio_codes ), args.output_path );

	vall_e_free( ctx );

	return 0;
}