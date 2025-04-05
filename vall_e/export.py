import argparse

import torch
import torch.nn

from .data import get_phone_symmap
from .engines import load_engines
from .config import cfg
from .models.lora import lora_get_state_dict
from .utils.io import torch_save, torch_load, json_read, json_write, Path

# stitches embeddings into one embedding & classifier => lm_head, for use in a HF compatible weight
# *will* require retraining because the classifier is in one contiguous space, and proms are NOT summed
@torch.no_grad()
def convert_to_hf_llama( state_dict, config = None, save_path = None ):
	n_text_tokens, model_dim = state_dict['module']['text_emb.weight'].shape

	n_audio_tokens = state_dict['module']['proms_emb.embeddings.0.weight'].shape[0]
	n_resp_levels = state_dict['module']['rvq_l_emb.weight'].shape[0]
	n_len_tokens = 11
	n_lang_tokens = state_dict['module']['langs_emb.weight'].shape[0]
	n_task_tokens = state_dict['module']['tasks_emb.weight'].shape[0]

	classifier_bias = "classifiers.proj.0.bias" in state_dict['module'] # cfg.model.experimental.classifiers_bias
	split_classifiers = "classifiers.proj.0.weight" in state_dict['module'] # cfg.model.experimental.split_classifiers

	# the new tokenizer to use
	tokenizer = {}
	tokenizer_vocab = {}

	tokenizer_path = cfg.rel_path / cfg.tokenizer_path
	if not tokenizer_path.exists():
		tokenizer_path = Path("./data/") / cfg.tokenizer_path
	if tokenizer_path.exists():
		tokenizer = json_read( tokenizer_path )
	else:
		tokenizer = {
			"model": {
				"vocab": get_phone_symmap()
			}
		}

	# cleanup duplicate IDs because convert_hf_to_gguf.py does not like this
	# get unique tokens
	for k, v in tokenizer["model"]["vocab"].items():
		if k not in tokenizer_vocab:
			tokenizer_vocab[k] = v
	# override if its in a merge
	for k, v in tokenizer["model"]["vocab"].items():
		for m in tokenizer["model"]["merges"]:
			if k in m:
				tokenizer_vocab[k] = v
				break
	tokenizer["model"]["vocab"] = {}


	lang_map = [
		"en",
		"ja",
		"de",
		"fr",
		"zh",
		"ko",
	]
	task_map = [
		"tts",
		"tts-c",
		"ns",
		"sr",
		"tse",
		"soe",
		"mask",
		"eoe",
		"stt",
	]
	tone_map = [
		"neutral",
	]

	# (start, end), embedding, classifier, token_format
	mapping = [
		[(0, 0), "text_emb.weight", "classifiers.proj.9.weight", None],
		[(0, 0), "rvq_l_emb.weight", None, "<|RVQ:{l}|>"],
		[(0, 0), "langs_emb.weight", None, "<|lang:{lang}|>"],
		[(0, 0), "tasks_emb.weight", None, "<|task:{task}|>"],
		[(0, 0), "len_emb.weight", "classifiers.proj.10.weight", "<|len:{id}|>"],
		[(0, 0), "tones_emb.weight", None, "<|tone:{tone}|>"],
		[(0, 0), "sep", None, "<|sep|>"],

		[(0, 0), "proms_emb.embeddings.0.weight", None, "<|P|0|{id}|>"],
		[(0, 0), "proms_emb.embeddings.1.weight", None, "<|P|1|{id}|>"],
		[(0, 0), "proms_emb.embeddings.2.weight", None, "<|P|2|{id}|>"],
		[(0, 0), "proms_emb.embeddings.3.weight", None, "<|P|3|{id}|>"],
		[(0, 0), "proms_emb.embeddings.4.weight", None, "<|P|4|{id}|>"],
		[(0, 0), "proms_emb.embeddings.5.weight", None, "<|P|5|{id}|>"],
		[(0, 0), "proms_emb.embeddings.6.weight", None, "<|P|6|{id}|>"],
		[(0, 0), "proms_emb.embeddings.7.weight", None, "<|P|7|{id}|>"],

		[(0, 0), "resps_emb.embeddings.0.weight", "classifiers.proj.0.weight", "<|R|AR|0:0|{id}|>"],
		[(0, 0), "resps_emb.embeddings.1.weight", "classifiers.proj.1.weight", "<|R|NAR|0:1|{id}|>"],
		[(0, 0), "resps_emb.embeddings.2.weight", "classifiers.proj.2.weight", "<|R|NAR|1:2|{id}|>"],
		[(0, 0), "resps_emb.embeddings.3.weight", "classifiers.proj.3.weight", "<|R|NAR|2:3|{id}|>"],
		[(0, 0), "resps_emb.embeddings.4.weight", "classifiers.proj.4.weight", "<|R|NAR|3:4|{id}|>"],
		[(0, 0), "resps_emb.embeddings.5.weight", "classifiers.proj.5.weight", "<|R|NAR|4:5|{id}|>"],
		[(0, 0), "resps_emb.embeddings.6.weight", "classifiers.proj.6.weight", "<|R|NAR|5:6|{id}|>"],
		[(0, 0), "resps_emb.embeddings.7.weight", "classifiers.proj.7.weight", "<|R|NAR|6:7|{id}|>"],
		[(0, 0), "resps_emb.embeddings.8.weight", "classifiers.proj.8.weight", "<|R|NAR|0:0|{id}|>"],
	]

	n_tokens = 0
	# to-do: figure out discrepancy
	for i, m in enumerate( mapping ):
		k_embd = mapping[i][1]
		embds = state_dict['module'][k_embd] if k_embd in state_dict['module'] else None

		n_tokens += 1 if embds.dim() == 1 else embds.shape[0]

	embedding = torch.nn.Embedding( n_tokens, model_dim )
	classifier = torch.nn.Linear( model_dim, n_tokens, bias=classifier_bias )

	if not split_classifiers:
		src = state_dict['module']['classifier.weight'][:]
		classifier.weight[:src.shape[0], ] = src

	# update ranges
	start = 0
	for i, m in enumerate( mapping ):
		# get previous start
		k_embd = mapping[i][1]
		k_head = mapping[i][2]
		token_format = mapping[i][3]

		embds = state_dict['module'][k_embd] if k_embd in state_dict['module'] else None
		head = state_dict['module'][k_head] if k_head in state_dict['module'] else None

		# expand if 1D
		if embds.dim() == 1:
			embds = embds.unsqueeze(0)

		tokens = embds.shape[0]

		if embds is not None:
			embedding.weight[start:start+tokens] = embds

		if split_classifiers and head is not None:
			classifier.weight[start:start+head.shape[0]] = head
		
		if token_format is not None:
			for idx in range(0, tokens):
				# RVQ level
				if "{l}" in token_format:
					token = token_format.format(l=idx)
				elif "{lang}" in token_format:
					token = token_format.format(lang=lang_map[idx])
				elif "{task}" in token_format:
					token = token_format.format(task=task_map[idx])
				elif "{tone}" in token_format:
					token = token_format.format(tone=tone_map[idx])
				elif "{id}" in token_format:
					token = token_format.format(id=idx)
				else:
					token = token_format
				tokenizer_vocab[token] = idx + start
		
		end = start + tokens
		mapping[i][0] = (start, end)
		start = end

	model_dict = {}
	# filter out the underlying model weights and extract them
	for k in state_dict['module'].keys():
		if not k.startswith('model.'):
			continue
		model_dict[k] = state_dict['module'][k].clone()

	embedding_dict = embedding.state_dict()
	classifier_dict = classifier.state_dict()
	model_dict['model.embed_tokens.weight'] = embedding_dict['weight']
	model_dict['lm_head.weight'] = classifier_dict['weight']
	if classifier_bias:
		model_dict['lm_head.bias'] = classifier_dict['bias']
	
	# write files in an HF compatible way
	out_dir = cfg.rel_path / "hf"
	out_dir.mkdir(parents=True, exist_ok=True)
	# write weights
	torch_save( { "module": model_dict, "format": "pt" }, out_dir / "model.safetensors" )
	# write tokenizer.json
	tokenizer['model']['vocab'] |= tokenizer_vocab
	json_write(tokenizer, out_dir / "tokenizer.json", pretty=True)
	# write tokenizer_config.json
	json_write({
  		"added_tokens": tokenizer['added_tokens'],
		"bos_token": "<bos>",
		"eos_token": "</eos>",
		"clean_up_tokenization_spaces": True,
		"model_input_names": [
			"input_ids",
			"attention_mask"
		],
		"tokenizer_class": "PreTrainedTokenizerFast"
	}, out_dir / "tokenizer_config.json", pretty=True)
	# write config.json
	json_write({
		"architectures": [
			"LLaMAForCausalLM"
		],
		"attention_bias": False,
		"attention_dropout": 0.0,
		"bos_token_id": 1,
		"eos_token_id": 2,
		"hidden_act": "gelu",
		"hidden_size": model_dim,
		"initializer_range": 0.02,
		"intermediate_size": model_dim * 4,
		"max_position_embeddings": 75 * 60 * 5,
		"model_type": "llama",
		"num_attention_heads": 16,
		"num_hidden_layers": 12,
		"num_key_value_heads": 16,
		"pretraining_tp": 1,
		"rms_norm_eps": 1e-06,
		"rope_scaling": None,
		"rope_theta": 10000.0,
		"tie_word_embeddings": False,
		"torch_dtype": "bfloat16",
		"transformers_version": "4.40.0",
		"use_cache": False,
		"vocab_size": n_tokens
	}, out_dir / "config.json", pretty=True )

	return state_dict

# yanks a LoRA from the training checkpoint
def extract_lora( state_dict, config = None, save_path = None, dtype = None ):
	if dtype is None:
		dtype = cfg.inference.dtype

	format = save_path.suffix[1:]

	lora = state_dict["lora"] if "lora" in state_dict else None
	# should always be included, but just in case
	if lora is None and "module" in state_dict:
		lora, module = lora_get_state_dict( state_dict["module"], split = True )
		state_dict["module"] = module
	
	if "lora" in state_dict:
		state_dict["lora"] = None

	# should raise an exception since there's nothing to extract, or at least a warning
	if not lora:
		return state_dict

	# save lora specifically
	# should probably export other attributes, similar to what SD LoRAs do
	save_path = save_path.parent / f"lora.{format}"
	torch_save( {
		"module": lora,
		"config": cfg.lora.__dict__ if cfg.lora is not None else None,
	}, save_path )

	return state_dict

# copies a single classifier head into multiple classifier heads per RVQ level
def split_classifier_heads( state_dict, config = cfg.model, save_path = None, dtype = None):
	levels = config.max_levels

	if "classifier.weight" not in state_dict['module']:
		return state_dict
	# copy to new AudioClassifier
	for i in range(levels):
		tokens = 1025 if i == 0 else 1024

		# trim per RVQ level (since level 0 has a stop token)
		state_dict['module'][f'classifiers.proj.{i}.weight'] = state_dict['module']['classifier.weight'][:tokens, :].clone()
		state_dict['module'][f'classifiers.proj.{i}.bias'] = state_dict['module']['classifier.bias'][:tokens].clone()

	# delete old weights
	del state_dict['module']['classifier.weight']
	del state_dict['module']['classifier.bias']

	return state_dict

# converts a normal LLaMA model to a MoE model, as best as I can
def moe_ify( state_dict, config = cfg.model, save_path = None, dtype = None ):
	# to-do: find a good way to pass in requested experts
	experts = 8
	for layer in range( config.layers ):
		#state_dict[f'model.layers.{layer}.block_sparse_moe.gate.weight'] = torch.randn((config.dim, experts))
		for expert in range( experts ):
			state_dict['module'][f'model.layers.{layer}.block_sparse_moe.experts.{expert}.w1.weight'] = state_dict['module'][f'model.layers.{layer}.mlp.up_proj.weight'].clone()
			state_dict['module'][f'model.layers.{layer}.block_sparse_moe.experts.{expert}.w2.weight'] = state_dict['module'][f'model.layers.{layer}.mlp.down_proj.weight'].clone()
			state_dict['module'][f'model.layers.{layer}.block_sparse_moe.experts.{expert}.w3.weight'] = state_dict['module'][f'model.layers.{layer}.mlp.gate_proj.weight'].clone()

		del state_dict['module'][f'model.layers.{layer}.mlp.up_proj.weight']
		del state_dict['module'][f'model.layers.{layer}.mlp.down_proj.weight']
		del state_dict['module'][f'model.layers.{layer}.mlp.gate_proj.weight']

	return state_dict

def main():
	parser = argparse.ArgumentParser("Save trained model to path.")
	parser.add_argument("--module-only", action='store_true')
	parser.add_argument("--hf", action='store_true', default=None) # convert to HF LLaMA
	parser.add_argument("--export-lora", action='store_true', default=None) # exports LoRA
	parser.add_argument("--split-classifiers", action='store_true', default=None) # splits classifier heads
	parser.add_argument("--moe-ify", action='store_true', default=None) # splits classifier heads
	parser.add_argument("--experts", type=int, default=8) # set target dtype to export to
	parser.add_argument("--dtype", type=str, default="auto") # set target dtype to export to
	parser.add_argument("--format", type=str, default=cfg.weights_format) # set target format to export weights under
	args, unknown = parser.parse_known_args()

	if args.format.lower() not in ["sft", "safetensors", "pt", "pth"]:
		raise Exception(f"Unknown requested format: {args.format}")

	if args.module_only:
		cfg.trainer.load_module_only = True


	if args.hf and args.export_lora:
		raise Exception("Requesting more than one callback")

	if args.dtype != "auto":
		cfg.trainer.weight_dtype = args.dtype
		
	# necessary to ensure we are actually exporting the weights right
	cfg.inference.backend = cfg.trainer.backend

	engines = load_engines(training=False) # to ignore loading optimizer state

	callback = None
	if args.hf:
		callback = convert_to_hf_llama
	elif args.export_lora:
		callback = extract_lora
	elif args.split_classifiers:
		callback = split_classifier_heads
	elif args.moe_ify:
		callback = moe_ify
		# set it here after the model loads to not influence which model loads
		cfg.model.experts = args.experts
		for name, engine in engines.items():
			engine.module.config.experts = args.experts
			engine.hyper_config.experts = args.experts

	engines.export(userdata={"symmap": get_phone_symmap()}, callback=callback, format=args.format)

if __name__ == "__main__":
	main()