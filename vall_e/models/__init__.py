import logging

import requests
from tqdm import tqdm
from pathlib import Path

_logger = logging.getLogger(__name__)

# to-do: implement automatically downloading model
DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / 'data/models'
DEFAULT_MODEL_URLS = {
	'ar+nar-tts+stt-llama-8/fp32.sft': 'https://huggingface.co/ecker/vall-e/resolve/main/models/ckpt/ar%2Bnar-tts%2Bstt-llama-8/fp32.sft',
}

# kludge, probably better to use HF's model downloader function
# to-do: write to a temp file then copy so downloads can be interrupted
def download_model( save_path, chunkSize = 1024, unit = "MiB" ):
	scale = 1
	if unit == "KiB":
		scale = (1024)
	elif unit == "MiB":
		scale = (1024 * 1024)
	elif unit == "MiB":
		scale = (1024 * 1024 * 1024)
	elif unit == "KB":
		scale = (1000)
	elif unit == "MB":
		scale = (1000 * 1000)
	elif unit == "MB":
		scale = (1000 * 1000 * 1000)

	name = save_path.name
	url = DEFAULT_MODEL_URLS[name] if name in DEFAULT_MODEL_URLS else None
	if url is None:
		raise Exception(f'Model requested for download but not defined: {name}')

	if not save_path.parent.exists():
		save_path.parent.mkdir(parents=True, exist_ok=True)

	r = requests.get(url, stream=True)
	content_length = int(r.headers['Content-Length'] if 'Content-Length' in r.headers else r.headers['content-length']) // scale

	with open(save_path, 'wb') as f:
		bar = tqdm( unit=unit, total=content_length )
		for chunk in r.iter_content(chunk_size=chunkSize): 
			if not chunk:
				continue
			
			bar.update( len(chunk) / scale )
			f.write(chunk)
		bar.close()


def get_model(config, training=True, **model_kwargs):
	name = config.name

	if "len" in config.capabilities:
		from .nar import NAR
		model = NAR(
			n_text_tokens=config.text_tokens,
			n_audio_tokens=config.audio_tokens,
			d_model=config.dim,
			n_heads=config.heads,
			n_layers=config.layers,
			n_experts=config.experts,
			
			p_dropout=config.dropout,
			
			l_padding = config.input_alignment,
			
			training = training,
			config = config,
			**model_kwargs
		)
	elif config.experimental.hf:
		from .experimental import Model as Experimental
		model = Experimental(
			n_text_tokens=config.text_tokens,
			n_audio_tokens=config.audio_tokens,

			d_model=config.dim,
			n_layers=config.layers,
			n_heads=config.heads,
			p_dropout=config.dropout,

			config = config,
			**model_kwargs
		)
	else:
		from .ar_nar import AR_NAR
		model = AR_NAR(
			n_text_tokens=config.text_tokens,
			n_audio_tokens=config.audio_tokens,
			d_model=config.dim,
			n_heads=config.heads,
			n_layers=config.layers,
			n_experts=config.experts,
			
			p_dropout=config.dropout,
			
			l_padding = config.input_alignment,
			
			training = training,
			config = config,
			**model_kwargs
		)

	_logger.info(f"{name} ({next(model.parameters()).dtype}): {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")

	return model

def get_models(models, training=True, **model_kwargs):
	return { model.full_name: get_model(model, training=training, **model_kwargs) for model in models }
