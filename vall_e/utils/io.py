import torch
import json

from pathlib import Path
from safetensors import safe_open as sft_load
from safetensors.torch import save_file as sft_save

try:
	use_orjson = True
	import orjson as json
except:
	import json

from .utils import truncate_json

def json_stringify( data, truncate=False, pretty=False, raw=False ):
	if truncate:
		return truncate_json( json.dumps( data ) )
	if pretty:
		s = json.dumps( data, option=json.OPT_INDENT_2 ) if use_orjson else json.dumps( data, indent='\t' )
		return s if raw and use_orjson else s.decode('utf-8')
	return json.dumps( data )

def json_parse( string ):
	return json.loads( string )

def json_read( path, default=None ):
	path = coerce_path( path )

	if not path.exists():
		return default

	with (open( str(path), "rb" ) if use_orjson else open( str(path), "r", encoding="utf-8" ) ) as f:
		return json_parse( f.read() )

def json_write( data, path, **kwargs ):
	path = coerce_path( path )
	
	with (open( str(path), "wb" ) if use_orjson else open( str(path), "w", encoding="utf-8" ) ) as f:
		f.write( json_stringify( data, raw=use_orjson, **kwargs ) )

def coerce_path( path ):
	return path if isinstance( path, Path ) else Path(path)

def pick_path( path, *suffixes ):
	suffixes = [*suffixes]

	for suffix in suffixes:
		p = path.with_suffix( suffix )
		if p.exists():
			return p

	return path

def is_dict_of( d, t ):
	if not isinstance( d, dict ):
		return False

	return all([ isinstance(v, torch.Tensor) for v in d.values() ])

# handles converting the usual pth state_dict into just the dict with the tensors + a dict of JSON strings, for safetensors
def state_dict_to_tensor_metadata( data: dict, module_key=None ):
	metadata = {}

	# is a state_dict, no need to coerce
	if is_dict_of( data, torch.Tensor ):
		return data, metadata

	# is maybe a dict with a state dict + metadata, coerce it
	target = module_key
	if not target:
		for k, v in data.items():
			# is a dict of tensors, our target
			if is_dict_of( v, torch.Tensor ):
				target = k
				continue # continue to iterate to grab other metadata

			# not a dict of tensors, put it as metadata
			try:
				metadata[k] = json_stringify(v) if any([isinstance( v, dict ), isinstance( v, list )]) else v

				if isinstance( metadata[k], bytes ):
					metadata[k] = metadata[k].decode('utf-8')
			except Exception as e:
				pass

	if not target:
		raise Exception(f'Requesting to save safetensors of a state dict, but state dict contains no key of torch.Tensor: {path}')

	return data[target], metadata

def torch_save( data, path, module_key=None ):
	path = coerce_path(path)
	ext = path.suffix

	if ext in [".safetensor", ".safetensors", ".sft"]:
		data, metadata = state_dict_to_tensor_metadata( data, module_key=module_key )

		if metadata is None:
			metadata = {}

		return sft_save( data, path, { k: v for k, v in metadata.items() if v is not None } )

	return torch.save( data, path )

def torch_load( path, device="cpu", framework="pt", unsafe=True, load_metadata=True, module_key="module" ):
	path = coerce_path(path)
	ext = path.suffix
	
	if ext in [".safetensor", ".safetensors", ".sft"]:
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