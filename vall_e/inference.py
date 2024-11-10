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
		#
		max_ar_steps=6 * cfg.dataset.frames_per_second,
		max_nar_levels=7,
		#
		input_prompt_length=0.0,
		input_prompt_prefix=False,
		prefix_silence=0.0,
		#
		ar_temp=0.0,
		nar_temp=0.0,
		#
		min_ar_temp=0.0,
		min_nar_temp=0.0,
		#
		top_p=1.0,
		top_k=0,
		min_p=0.0,
		#
		repetition_penalty=1.0,
		repetition_penalty_decay=0.0,
		length_penalty=0.0,
		#
		beam_width=0,
		#
		mirostat_tau=0,
		mirostat_eta=0.1,
		#
		dry_multiplier=0.0,
		dry_base=1.75,
		dry_allowed_length=2,
		#
		entropix_sampling=False,
		#
		layer_skip=False,
		layer_skip_exit_layer=-1,
		layer_skip_entropy_threshold=-1,
		layer_skip_varentropy_threshold=-1,
		#
		refine_on_stop=False,
		#
		seed = None,
		#
		load_from_artifact = None,
		denoise_start = 0.0,

		out_path=None,

		tqdm=True,
		use_lora=None,
	):
		lines = text.split("\n")

		wavs = []
		sr = None

		model_ar = None
		model_len = None
		model_nar = None

		for name, engine in self.engines.items():
			if "ar" in engine.hyper_config.capabilities:
				model_ar = engine.module
			if "len" in engine.hyper_config.capabilities:
				model_len = engine.module
			if "nar" in engine.hyper_config.capabilities:
				model_nar = engine.module
		
		seed = set_seed(seed)

		if task == "stt":
			resp = self.encode_audio( references )
			lang = self.encode_lang( language )
			
			resp = to_device(resp, device=self.device, dtype=torch.int16)
			lang = to_device(lang, device=self.device, dtype=torch.uint8)

			with torch.autocast("cuda", dtype=self.dtype, enabled=self.amp):
				if model_ar is not None:
					text_list = model_ar(
						text_list=None, proms_list=[resp], lang_list=[lang], resps_list=[resp], max_steps=max_ar_steps, task_list=["stt"],
						sampling_temperature=ar_temp,
						sampling_min_temperature=min_ar_temp,
						sampling_top_p=top_p, sampling_top_k=top_k, sampling_min_p=min_p,
						sampling_repetition_penalty=repetition_penalty, sampling_repetition_penalty_decay=repetition_penalty_decay,
						sampling_length_penalty=length_penalty,
						sampling_beam_width=beam_width,
						sampling_mirostat_tau=mirostat_tau,
						sampling_mirostat_eta=mirostat_eta,
						sampling_dry_multiplier=dry_multiplier,
						sampling_dry_base=dry_base,
						sampling_dry_allowed_length=dry_allowed_length,
						sampling_entropix=entropix_sampling,
						sampling_layer_skip=layer_skip,
						sampling_layer_skip_exit_layer=layer_skip_exit_layer,
						sampling_refine_on_stop=refine_on_stop,

						disable_tqdm=not tqdm,
						use_lora=use_lora,
					)
				else:
					raise Exception("!")
				
				text_list = [ cfg.tokenizer.decode( text ).replace("   ", "_").replace(" ", "").replace("_", " ") for text in text_list ]

			return text_list[0]

		# validate settings here
		if not references and ar_temp < 0.5:
			_logger.warning(f'Audio-promptless inferencing fails with low AR temperatures.')

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
				if model_ar is not None:
					resps_list = model_ar(
						text_list=[phns], proms_list=[prom], lang_list=[lang], max_steps=max_ar_steps, task_list=["tts"],
						input_prompt_prefix=input_prompt_prefix,
						prefix_silence=prefix_silence,
						sampling_temperature=ar_temp,
						sampling_min_temperature=min_ar_temp,
						sampling_top_p=top_p, sampling_top_k=top_k, sampling_min_p=min_p,
						sampling_repetition_penalty=repetition_penalty, sampling_repetition_penalty_decay=repetition_penalty_decay,
						sampling_length_penalty=length_penalty,
						sampling_beam_width=beam_width,
						sampling_mirostat_tau=mirostat_tau,
						sampling_mirostat_eta=mirostat_eta,
						sampling_dry_multiplier=dry_multiplier,
						sampling_dry_base=dry_base,
						sampling_dry_allowed_length=dry_allowed_length,
						sampling_entropix=entropix_sampling,
						sampling_layer_skip=layer_skip,
						sampling_layer_skip_exit_layer=layer_skip_exit_layer,
						sampling_layer_skip_entropy_threshold=layer_skip_entropy_threshold,
						sampling_layer_skip_varentropy_threshold=layer_skip_varentropy_threshold,
						sampling_refine_on_stop=refine_on_stop,

						disable_tqdm=not tqdm,
						use_lora=use_lora,
					)
					resps_list = model_nar(
						text_list=[phns], proms_list=[prom], lang_list=[lang], resps_list=resps_list, task_list=["tts"],
						input_prompt_prefix=input_prompt_prefix,
						max_levels=max_nar_levels,
						sampling_temperature=nar_temp,
						sampling_min_temperature=min_nar_temp,
						sampling_top_p=top_p, sampling_top_k=top_k, sampling_min_p=min_p,
						sampling_repetition_penalty=repetition_penalty, sampling_repetition_penalty_decay=repetition_penalty_decay,
						sampling_layer_skip=layer_skip,
						sampling_layer_skip_exit_layer=layer_skip_exit_layer,
						sampling_layer_skip_entropy_threshold=layer_skip_entropy_threshold,
						sampling_layer_skip_varentropy_threshold=layer_skip_varentropy_threshold,

						disable_tqdm=not tqdm,
						use_lora=use_lora,
					)
				elif model_len is not None:
					len_list = model_len( text_list=[phns], proms_list=[prom], task_list=["len"], max_steps=5, disable_tqdm=not tqdm ) # don't need more than that
					len_list = [ clamp(l, 1, max_ar_steps) for l in len_list ]
					
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
						max_steps=max_ar_steps,
						max_levels=max_nar_levels,
						sampling_temperature=nar_temp,
						sampling_min_temperature=min_nar_temp,
						sampling_top_p=top_p, sampling_top_k=top_k, sampling_min_p=min_p,
						sampling_repetition_penalty=repetition_penalty, sampling_repetition_penalty_decay=repetition_penalty_decay,
						denoise_start=denoise_start,

						disable_tqdm=not tqdm,
						use_lora=use_lora,
						**kwargs,
					)
				else:
					raise Exception("!")

			wav, sr = qnt.decode_to_file(resps_list[0], out_path, device=self.device)
			wavs.append(wav)
		
		return (torch.concat(wavs, dim=-1), sr)
		
