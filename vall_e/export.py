import argparse

import torch
import torch.nn

from .data import get_phone_symmap
from .engines import load_engines
from .config import cfg
from .models.lora import lora_get_state_dict
from .utils.io import torch_save, torch_load

# stitches embeddings into one embedding & classifier => lm_head, for use in a HF compatible weight
# *will* require retraining because the classifier is in one contiguous space, and proms are NOT summed
@torch.no_grad()
def convert_to_hf( state_dict, config = None, save_path = None ):
	# to-do: infer all of this from the existing state_dict, should be easy by checking shape
	model_dim = 1024

	n_text_tokens = 256
	n_audio_tokens = 1024
	n_resp_levels = 8
	n_len_tokens = 11
	n_lang_tokens = 4
	n_task_tokens = 9

	# the new tokenizer to use
	tokenizer_append = {}

	l_tokens = [
		n_text_tokens, # text
		n_audio_tokens * n_resp_levels, # prom
		(n_audio_tokens + 1) * 2, # resp: AR + NAR-len (with stop/mask)
		(n_audio_tokens) * (n_resp_levels - 1), # NAR
		n_resp_levels, # RVQ level
		n_len_tokens, # len tokens
		1, # separator
		n_lang_tokens, # langs
		n_task_tokens, # tasks
	]

	n_tokens = sum(l_tokens)

	lang_map = [
		"en",
		"ja",
		"de",
		"fr",
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

	embedding = torch.nn.Embedding( n_tokens, model_dim )
	classifier = torch.nn.Linear( model_dim, n_tokens )

	#embedding.weight.requires_grad = False
	#classifier.weight.requires_grad = False
	#classifier.bias.requires_grad = False

	# inject text tokens
	token_start = 0
	token_end = l_tokens[0]
	embedding.weight[token_start:token_end] = state_dict['module']['text_emb.weight']
	classifier.weight[token_start:token_end] = state_dict['module']['classifiers.proj.9.weight']
	classifier.bias[token_start:token_end] = state_dict['module']['classifiers.proj.9.bias']
	# tokenizer already has these tokens

	# inject prom tokens
	token_start = token_end
	token_end += l_tokens[1]
	for l in range(n_resp_levels):
		start = token_start + (l*n_resp_levels)
		end = start + n_audio_tokens
		embedding.weight[start:end] = state_dict['module'][f'proms_emb.embeddings.{l}.weight']
		# there's no corresponding classifier
		#classifier.weight[start:end] = state_dict['module'][f'classifiers.proj.{l}.weight']
		#classifier.bias[start:end] = state_dict['module'][f'classifiers.proj.{l}.bias']
		for t in range(n_audio_tokens):
			tokenizer_append[f'<P:{l}:{t}>'] = start + t

	# inject AR
	token_start = token_end
	token_end += l_tokens[2] // 2
	embedding.weight[token_start:token_end] = state_dict['module'][f'resps_emb.embeddings.0.weight']
	classifier.weight[token_start:token_end] = state_dict['module']['classifiers.proj.0.weight']
	classifier.bias[token_start:token_end] = state_dict['module']['classifiers.proj.0.bias']
	for t in range(n_audio_tokens):
		tokenizer_append[f'<AR:0:0:{t}>'] = token_start + t
	tokenizer_append[f'<AR:0:0:STOP>'] = token_start + 1024

	# inject NAR-len
	token_start = token_end
	token_end += l_tokens[2] // 2
	embedding.weight[token_start:token_end] = state_dict['module'][f'resps_emb.embeddings.8.weight']
	classifier.weight[token_start:token_end] = state_dict['module']['classifiers.proj.8.weight']
	classifier.bias[token_start:token_end] = state_dict['module']['classifiers.proj.8.bias']
	for t in range(n_audio_tokens):
		tokenizer_append[f'<NAR:0:0:{t}>'] = token_start + t
	tokenizer_append[f'<NAR:0:0:STOP>'] = token_start + 1024
	
	# inject NAR
	token_start = token_end
	token_end += l_tokens[3]
	for l in range(1, n_resp_levels):
		start = token_start + ((l-1)*n_resp_levels)
		end = start + n_audio_tokens
		embedding.weight[start:end] = state_dict['module'][f'resps_emb.embeddings.{l}.weight']
		classifier.weight[start:end] = state_dict['module'][f'classifiers.proj.{l}.weight']
		classifier.bias[start:end] = state_dict['module'][f'classifiers.proj.{l}.bias']
		for t in range(n_audio_tokens):
			tokenizer_append[f'<NAR:{l-1}:{l}:{t}>'] = start + t
	
	# inject RVQ level
	token_start = token_end
	token_end += l_tokens[4]
	embedding.weight[token_start:token_end] = state_dict['module'][f'rvq_l_emb.weight']
	# there is no corresponding classifier
	for l in range(n_resp_levels):
		tokenizer_append[f'<RVQ:{l}>'] = token_start + l

	# inject len
	token_start = token_end
	token_end += l_tokens[5]
	embedding.weight[token_start:token_end] = state_dict['module'][f'len_emb.weight']
	classifier.weight[token_start:token_end] = state_dict['module']['classifiers.proj.10.weight'][0:n_len_tokens] # erroneously sized as 256
	classifier.bias[token_start:token_end] = state_dict['module']['classifiers.proj.10.bias'][0:n_len_tokens] # erroneously sized as 256
	for t in range(n_len_tokens):
		tokenizer_append[f'<len:{t}>'] = token_start + t

	# inject sep
	token_start = token_end
	token_end += l_tokens[6]
	embedding.weight[token_start:token_end] = state_dict['module']['sep']
	tokenizer_append['<sep>'] = token_start
	# there is no corresponding classifier

	# inject langs
	token_start = token_end
	token_end += l_tokens[7]
	embedding.weight[token_start:token_end] = state_dict['module']['langs_emb.weight']
	for l in range(n_lang_tokens):
		lang = lang_map[l]
		tokenizer_append[f'<lang:{lang}>'] = token_start + l
	# there is no corresponding classifier

	# inject tasks
	token_start = token_end
	token_end += l_tokens[8]
	embedding.weight[token_start:token_end] = state_dict['module']['tasks_emb.weight']
	for l in range(n_task_tokens):
		task = task_map[l]
		tokenizer_append[f'<task:{task}>'] = token_start + l
	# there is no corresponding classifier


	model_dict = {}
	# filter out the underlying model weights and extract them
	for k in state_dict['module'].keys():
		if not k.startswith('model.'):
			continue
		model_dict[k] = state_dict['module'][k].clone()
	del state_dict['module']

	embedding_dict = embedding.state_dict()
	classifier_dict = classifier.state_dict()
	model_dict['model.embed_tokens.weight'] = embedding_dict['weight']
	model_dict['lm_head.weight'] = classifier_dict['weight']
	model_dict['lm_head.bias'] = classifier_dict['bias']

	state_dict['module'] = model_dict
	state_dict['vocab'] = tokenizer_append

	return state_dict

	"""
	n_tokens = 256 + (1024 * 8) + (1024 * 8) + 1
	token_dim = 1024
	embedding = torch.nn.Embedding(n_tokens, token_dim)
	embedding.weight.requires_grad = False

	def move_value(k):
		v = state_dict['module'][k]
		del state_dict['module'][k]
		return v

	separator = move_value('sep')
	out_proj = move_value('classifier.weight')
	text_emb = move_value('text_emb.weight')
	langs_emb = move_value('langs_emb.weight')
	tasks_emb = move_value('tasks_emb.weight')
	tones_emb = move_value('tones_emb.weight')
	
	proms_emb_weight = [ move_value(f'proms_emb.weight.{i}').item() for i in range(8) ] if "proms_emb.weight.0" in state_dict['module'] else [ [ 1 for _ in range(8) ] ]
	resps_emb_weight = [ move_value(f'resps_emb.weight.{i}').item() for i in range(8) ] if "resps_emb.weight.0" in state_dict['module'] else [ [ 1 for _ in range(8) ] ]

	proms_emb = [ move_value(f'proms_emb.embeddings.{i}.weight') for i in range(8) ]
	resps_emb = [ move_value(f'resps_emb.embeddings.{i}.weight') for i in range(8) ]


	start = 0
	for i in range(256):
		embedding.weight[start + i] = text_emb[i]
	
	start = 256
	for layer in range(8):
		for i in range(1024):
			offset = start + 1024 * layer
			embedding.weight[i + offset] = proms_emb[layer][i] * proms_emb_weight[layer]
			
	start = 256 + 1024 * 8
	for layer in range(8):
		for i in range(1024):
			offset = start + 1024 * layer
			embedding.weight[i + offset] = resps_emb[layer][i] * proms_emb_weight[layer]

	state_dict['module']['model.embed_tokens.weight'] = embedding.state_dict()
	# to-do: properly recreate the output head weights or something
	state_dict['module']['lm_head.weight'] = out_proj
	
	del state_dict['module']['classifier.weight']
	del state_dict['module']['classifier.bias']

	return state_dict
	"""

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
	parser.add_argument("--hf", action='store_true', default=None) # convert to HF-style
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
		callback = convert_to_hf
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