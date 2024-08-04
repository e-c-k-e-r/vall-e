import torch
import json

from pathlib import Path
from safetensors import safe_open as sft_load
from safetensors.torch import save_file as sft_save

def coerce_path( path ):
	return path if isinstance( path, Path ) else Path(path)

def is_dict_of( d, t ):
	if not isinstance( d, dict ):
		return False

	return all([ isinstance(v, torch.Tensor) for v in d.values() ])

# handles converting the usual pth state_dict into just the dict with the tensors + a dict of JSON strings, for safetensors
def state_dict_to_tensor_metadata( data: dict, module_key=None ):
	metadata = None

	# is a state_dict, no need to coerce
	if is_dict_of( data, torch.Tensor ):
		return data, metadata

	# is maybe a dict with a state dict + metadata, coerce it
	metadata = {}
	target = module_key
	if not target:
		for k, v in data.items():
			# is a dict of tensors, our target
			if is_dict_of( v, torch.Tensor ):
				target = k
				continue # continue to iterate to grab other metadata

			# not a dict of tensors, put it as metadata
			try:
				metadata[k] = json.dumps(v)
			except Exception as e:
				pass

	if not target:
		raise Exception(f'Requesting to save safetensors of a state dict, but state dict contains no key of torch.Tensor: {path}')

	return data[target], metadata

def torch_save( data, path, module_key=None ):
	path = coerce_path(path)
	ext = path.suffix

	if ext in [".safetensor", ".sft"]:
		data, metadata = state_dict_to_tensor_metadata( data, module_key=module_key )

		return sft_save( data, path, metadata )

	return torch.save( data, path )

def torch_load( path, device="cpu", framework="pt", unsafe=True, load_metadata=True, module_key="module" ):
	path = coerce_path(path)
	ext = path.suffix
	
	if ext in [".safetensor", ".sft"]:
		state_dict = {}
		with sft_load(path, framework=framework, device=device) as f:
			for k in f.keys():
				state_dict[k] = f.get_tensor(k)

			if load_metadata:
				metadata = f.metadata()
				for k, v in metadata.items():
					try:
						metadata[k] = json.loads( v )
					except Exception as e:
						pass
				state_dict = { module_key: state_dict } | metadata

		return state_dict

	return torch.load( path, map_location=torch.device(device), weights_only=not unsafe )