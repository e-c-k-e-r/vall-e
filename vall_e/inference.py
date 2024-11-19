import torch
import torchaudio
import soundfile
import time
import logging
import numpy as np

_logger = logging.getLogger(__name__)

from torch import Tensor
from einops import rearrange
from pathlib import Path

from .emb import g2p, qnt
from .emb.qnt import trim, trim_random, unload_model, repeat_extend_audio
from .utils import to_device, set_seed, clamp, wrapper as ml

from .config import cfg, Config
from .models import get_models
from .models.lora import enable_lora
from .engines import load_engines, deepspeed_available
from .data import get_phone_symmap, get_lang_symmap, _load_quants, _cleanup_phones, tokenize
from .models import download_model, DEFAULT_MODEL_PATH

if deepspeed_available:
	import deepspeed

class TTS():
	def __init__( self, config=None, lora=None, device=None, amp=None, dtype=None, attention=None ):
		self.loading = True 

		# yes I can just grab **kwargs and forward them here
		self.load_config( config=config, lora=lora, device=device, amp=amp, dtype=dtype, attention=attention )	
		self.load_model()

		self.loading = False 

	def load_config( self, config=None, lora=None, device=None, amp=None, dtype=None, attention=None ):
		if not config:
			download_model()
			config = DEFAULT_MODEL_PATH

		if config.suffix == ".yaml":
			_logger.info(f"Loading YAML: {config}")
			cfg.load_yaml( config )
		elif config.suffix == ".sft":
			_logger.info(f"Loading model: {config}")
			cfg.load_model( config, lora )
		else:
			raise Exception(f"Unknown config passed: {config}")		

		cfg.format( training=False )
		cfg.dataset.use_hdf5 = False # could use cfg.load_hdf5(), but why would it ever need to be loaded for inferencing

		if amp is None:
			amp = cfg.inference.amp
		if dtype is None or dtype == "auto":
			dtype = cfg.inference.weight_dtype
		if device is None:
			device = cfg.device

		cfg.device = device
		cfg.mode = "inferencing"
		cfg.trainer.backend = cfg.inference.backend
		cfg.trainer.weight_dtype = dtype
		cfg.inference.weight_dtype = dtype

		self.device = device
		self.dtype = cfg.inference.dtype
		self.amp = amp
		
		self.model_kwargs = {}
		if attention:
			self.model_kwargs["attention"] = attention

	def load_model( self ):
		load_engines.cache_clear()
		unload_model()
		
		self.engines = load_engines(training=False, **self.model_kwargs)
		for name, engine in self.engines.items():
			if self.dtype != torch.int8:
				engine.to(self.device, dtype=self.dtype if not self.amp else torch.float32)

		self.engines.eval()
		self.symmap = get_phone_symmap()
		_logger.info("Loaded model")

	def enable_lora( self, enabled=True ):
		for name, engine in self.engines.items():
			enable_lora( engine.module, mode = enabled )

	def disable_lora( self ):
		return self.enable_lora( enabled=False )

	def encode_text( self, text, language="en" ):
		# already a tensor, return it
		if isinstance( text, Tensor ):
			return text

		content = g2p.encode(text, language=language)
		tokens = tokenize( content )

		return torch.tensor( tokens )

	def encode_lang( self, language ):
		symmap = get_lang_symmap()
		id = 0
		if language in symmap:
			id = symmap[language]
		return torch.tensor([ id ])

	# to-do: trim before quantizing, instead of after
	def encode_audio( self, paths, trim_length=5.0 ):
		# already a tensor, return it
		if isinstance( paths, Tensor ):
			return paths

		# split string into paths
		if isinstance( paths, str ):
			paths = [ Path(p) for p in paths.split(";") ]

		# merge inputs

		proms = []

		for path in paths:
			prom = qnt.encode_from_file(path)
			if hasattr( prom, "codes" ):
				prom = prom.codes
			prom = prom[0][:, :].t().to(torch.int16)

			proms.append( prom )

		res = torch.cat(proms)
		
		if trim_length:
			res = repeat_extend_audio( res, int( cfg.dataset.frames_per_second * trim_length ) )
			#res = trim( res, int( cfg.dataset.frames_per_second * trim_length ) )
		
		return res

	@torch.inference_mode()
	def text_embedding( self, input, prom=False ):
		model = None

		for name, engine in self.engines.items():
			model = engine.module
			break

		if isinstance( input, str ):
			input = cfg.tokenizer.encode(input)

		if isinstance( input, list ):
			input = torch.tensor( input, dtype=torch.uint8, device=self.device )

		return model.text_emb( input )

	@torch.inference_mode()
	def audio_embedding( self, input, prom=False ):
		model = None

		for name, engine in self.engines.items():
			model = engine.module
			break

		# im really not sure which way is the better way, since the proms_emb and resps_emb have different properties.......
		if prom:
			return model.proms_emb(
				input,
				quant_level=input.shape[-1] - 1,
				offset=0,
				sums=True,
			)
		return sum([ model.resps_emb(
			input[:, :l+1],
			offset = 0 if l == 0 else 1, # or maybe set to 1
			quant_level = l,
			sums = False
		) for l in range( input.shape[-1] - 1 ) ])

	@torch.inference_mode()
	def inference(
		self,
		text,
		references,
		language="en",
		task="tts",

		input_prompt_length = 0,
		load_from_artifact = False,

		seed = None,
		out_path=None,
		tqdm=True,
		use_lora=None,
		**sampling_kwargs,
	):
		lines = text.split("\n")

		wavs = []
		sr = None

		model_ar = None
		model_len = None
		model_nar = None

		for name, engine in self.engines.items():
			if model_ar is None and "ar" in engine.hyper_config.capabilities:
				model_ar = engine.module
			if model_len is None and "len" in engine.hyper_config.capabilities:
				model_len = engine.module
			if model_nar is None and "nar" in engine.hyper_config.capabilities:
				model_nar = engine.module
		
		seed = set_seed(seed)

		if task == "stt":
			resp = self.encode_audio( references )
			lang = self.encode_lang( language )
			
			resp = to_device(resp, device=self.device, dtype=torch.int16)
			lang = to_device(lang, device=self.device, dtype=torch.uint8)

			with torch.autocast("cuda", dtype=self.dtype, enabled=self.amp):
				model = model_ar if model_ar is not None else model_nar
				if model is not None:
					text_list = model(
						text_list=None, proms_list=[resp], lang_list=[lang], resps_list=[resp], task_list=["stt"],
						disable_tqdm=not tqdm,
						use_lora=use_lora,
						**sampling_kwargs,
					)
				else:
					raise Exception("!")
				
				text_list = [ cfg.tokenizer.decode( text ).replace("   ", "_").replace(" ", "").replace("_", " ") for text in text_list ]

			return text_list[0]

		for line in lines:
			if out_path is None:
				output_dir = Path("./data/results/")
				if not output_dir.exists():
					output_dir.mkdir(parents=True, exist_ok=True)
				out_path = output_dir / f"{time.time()}.wav"

			prom = self.encode_audio( references, trim_length=input_prompt_length ) if references else None
			phns = self.encode_text( line, language=language )
			lang = self.encode_lang( language )

			prom = to_device(prom, device=self.device, dtype=torch.int16)
			phns = to_device(phns, device=self.device, dtype=torch.uint8 if len(self.symmap) < 256 else torch.int16)
			lang = to_device(lang, device=self.device, dtype=torch.uint8)

			# to-do: add in case for experimental.hf model
			with torch.autocast("cuda", dtype=self.dtype, enabled=self.amp):
				if model_len is not None:
					len_list = model_len( text_list=[phns], proms_list=[prom], task_list=["len"], disable_tqdm=not tqdm, **{"max_steps": 5} ) # don't need more than that
					kwargs = {}
					# nasty hardcode to load a reference file and have that as the input target
					if load_from_artifact and load_from_artifact.exists():
						artifact = np.load(load_from_artifact, allow_pickle=True)[()]

						phns = torch.tensor( cfg.tokenizer.encode( artifact["metadata"]["phonemes"] ) ).to(dtype=torch.uint8, device=self.device)
						resp = torch.from_numpy(artifact["codes"].astype(np.int16))[0, :, :].t().to(dtype=torch.int16, device=self.device)
						prom = resp[:75*3, :]
						len_list = [ resp.shape[0] ]

						kwargs["resps_list"] = [ resp[:, :1] ]

					resps_list = model_nar( text_list=[phns], proms_list=[prom], len_list=len_list, task_list=["tts"],
						disable_tqdm=not tqdm,
						use_lora=use_lora,
						**(sampling_kwargs | kwargs),
					)
				elif model_ar is not None:
					resps_list = model_ar(
						text_list=[phns], proms_list=[prom], lang_list=[lang], task_list=["tts"],
						disable_tqdm=not tqdm,
						use_lora=use_lora,
						**sampling_kwargs,
					)
					resps_list = model_nar(
						text_list=[phns], proms_list=[prom], lang_list=[lang], resps_list=resps_list, task_list=["tts"],
						disable_tqdm=not tqdm,
						use_lora=use_lora,
						**sampling_kwargs,
					)
				else:
					raise Exception("!")

			wav, sr = qnt.decode_to_file(resps_list[0], out_path, device=self.device)
			wavs.append(wav)
		
		return (torch.concat(wavs, dim=-1), sr)
		
