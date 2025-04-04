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

from tqdm import tqdm, trange

from .emb import g2p, qnt
from .emb.qnt import trim, trim_random, unload_model, repeat_extend_audio, AVAILABLE_AUDIO_BACKENDS
from .emb.transcribe import transcribe

from .utils import to_device, set_seed, clamp, ml

from .config import cfg, Config
from .models import get_models
from .models.lora import enable_lora
from .engines import load_engines, deepspeed_available
from .data import get_phone_symmap, get_lang_symmap, tokenize, text_tokenize, sentence_split
from .models import download_model, DEFAULT_MODEL_PATH

if deepspeed_available:
	import deepspeed

try:
	import sounddevice as sd
except Exception as e:
	sd = None

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

		# fallback to encodec if no vocos
		if cfg.audio_backend == "vocos" and "vocos" not in AVAILABLE_AUDIO_BACKENDS:
			_logger.warning("Vocos requested but not available, falling back to Encodec...")
			cfg.set_audio_backend(cfg.audio_backend)

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
		self.batch_size = cfg.inference.batch_size
		
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

	def encode_text( self, text, language="auto", precheck=True, phonemize=True ):
		# already a tensor, return it
		if isinstance( text, Tensor ):
			return text

		# check if tokenizes without any unks (for example, if already phonemized text is passes)
		if precheck and "<unk>" in self.symmap:
			tokens = tokenize( text )
			if self.symmap["<unk>"] not in tokens:
				return torch.tensor( tokens )

		if not phonemize:
			return torch.tensor( text_tokenize( text ) )

		return torch.tensor( tokenize( g2p.encode(text, language=language) ) )

	def encode_lang( self, language ):
		symmap = get_lang_symmap()
		id = 0
		if language in symmap:
			id = symmap[language]
		return torch.tensor([ id ])

	# to-do: trim before quantizing, instead of after
	def encode_audio( self, paths, trim_length=0.0 ):
		# already a tensor, return it
		if isinstance( paths, Tensor ):
			return paths

		# split string into paths
		if isinstance( paths, str ):
			paths = [ Path(p) for p in paths.split(";") ]

		# not already a list		
		if isinstance( paths, Path ):
			paths = [ paths ]

		proms = []

		# merge inputs
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

	def modality( self, modality ):
		# cringe to handle the best default mode for a given model
		if modality == "auto" and cfg.model.name in ["ar+nar", "nar-len"]:
			modality = cfg.model.name
		return modality

	# makes use of being able to batch inputs seamlessly by automatically batching
	# this is NOT the default because it absolutely cannot make use of rolling context / prefixing
	@torch.inference_mode()
	def batched_inference(
		self,
		texts,
		references=None,
		languages=None,
		text_languages=None,
		out_paths=None,
		**sampling_kwargs,
	):
		batch_size = sampling_kwargs.pop("batch_size", self.batch_size)
		input_prompt_length = sampling_kwargs.pop("input_prompt_length", 0)
		modality = sampling_kwargs.pop("modality", "auto")
		seed = sampling_kwargs.pop("seed", None)
		use_tqdm = sampling_kwargs.pop("tqdm", True)
		use_lora = sampling_kwargs.pop("use_lora", None)
		dtype = sampling_kwargs.pop("dtype", self.dtype)
		amp = sampling_kwargs.pop("amp", self.amp)

		if batch_size < 1:
			batch_size = 1

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
		
		modality = self.modality( modality )
		# force AR+NAR
		if modality == "ar+nar":
			model_len = None
		# force NAR-len
		elif modality == "nar-len":
			model_ar = None

		samples = len(texts)
		# fill with null input proms
		if not references:
			references = [ None for _ in range(samples) ]
		# fill with english
		if not languages:
			languages = [ "auto" for _ in range(samples) ]
		if not out_paths:
			out_paths = [ None for _ in range(samples) ]
		# use the audio language to phonemize the text
		if not text_languages:
			text_languages = languages

		inputs = []
		# tensorfy inputs
		for i in trange( samples, desc="Preparing batches" ):
			# detect language 
			if languages[i] == "auto":
				languages[i] = g2p.detect_language( texts[i] )

			texts[i] = self.encode_text( texts[i], language=text_languages[i] )
			references[i] = self.encode_audio( references[i], trim_length=input_prompt_length ) if references[i] else None
			languages[i] = self.encode_lang( languages[i] )

			texts[i] = to_device(texts[i], device=self.device, dtype=torch.uint8 if len(self.symmap) < 256 else torch.int16)
			references[i] = to_device(references[i], device=self.device, dtype=torch.int16)
			languages[i] = to_device(languages[i], device=self.device, dtype=torch.uint8)

			seq_len = texts[i].shape[0] + 1 + (references[i].shape[0] if references[i] is not None else 0) + 1

			inputs.append((texts[i], references[i], languages[i], out_paths[i], seq_len))

		# attempt to reduce padding
		inputs.sort(key=lambda x: x[-1])

		# create batches
		batches = []
		buffer = ([], [], [], [])
		for batch in inputs:
			# flush
			if len(buffer[0]) >= batch_size:
				batches.append(buffer)
				buffer = ([], [], [], [])

			# insert into buffer
			for i, x in enumerate( batch[:-1] ):
				buffer[i].append(x)

		# flush
		if buffer:
			batches.append(buffer)
			buffer = ([], [], [], [])

		wavs = []
		for texts, proms, langs, out_paths in tqdm(batches, desc="Processing batch"):
			seed = set_seed(seed)
			batch_size = len(texts)
			input_kwargs = dict(
				phns_list=texts,
				proms_list=proms,
				lang_list=langs,
				disable_tqdm=not use_tqdm,
				use_lora=use_lora,
			)

			with torch.autocast(self.device, dtype=dtype, enabled=amp):
				if model_len is not None:
					# extra kwargs
					duration_padding = sampling_kwargs.pop("duration_padding", 1.05)
					len_list = model_len( **input_kwargs, task_list=["len"]*batch_size, **{"max_duration": 5} ) # "max_duration" is max tokens

					# add an additional X seconds
					len_list = [ int(l * duration_padding) for l in len_list ]

					resps_list = model_nar( **input_kwargs, len_list=len_list, task_list=["tts"]*batch_size,
						**sampling_kwargs,
					)
				elif model_ar is not None:
					resps_list = model_ar(
						**input_kwargs, task_list=["tts"]*batch_size,
						**sampling_kwargs,
					)

					resps_list = model_nar(
						**input_kwargs, resps_list=resps_list, task_list=["tts"]*batch_size,
						**sampling_kwargs,
					)
				else:
					raise Exception("!")

				for resp, out_path in zip( resps_list, out_paths ):
					if out_path:
						wav, sr = qnt.decode_to_file(resp, out_path, device=self.device)
					else:
						wav, sr = qnt.decode(resp, device=self.device)
					wavs.append(wav)
		return wavs

	# naive serial inferencing
	# will automatically split a text into pieces (if requested) piece by piece
	@torch.inference_mode()
	def inference(
		self,
		text,
		references,
		language="auto",
		text_language=None,
		task="tts",
		out_path=None,
		play=False,
		**sampling_kwargs,
	):
		if sd is None:
			play = False

		input_prompt_length = sampling_kwargs.pop("input_prompt_length", 0)
		modality = sampling_kwargs.pop("modality", "auto")
		seed = sampling_kwargs.pop("seed", None)
		use_tqdm = sampling_kwargs.pop("tqdm", True)
		use_lora = sampling_kwargs.pop("use_lora", None)
		dtype = sampling_kwargs.pop("dtype", self.dtype)
		amp = sampling_kwargs.pop("amp", self.amp)
		phonemize = sampling_kwargs.pop("phonemize", True)
		duration_padding = sampling_kwargs.pop("duration_padding", 1.05)

		voice_convert = sampling_kwargs.pop("voice_convert", None)
		# explicitly require this
		if task != "vc":
			voice_convert = None
		elif voice_convert == None:
			raise Exception("Voice conversion requested, but no reference clip provided.")

		# transcribe from audio to voice convert from
		if voice_convert is not None and not text:
			text = transcribe( voice_convert, model_name="openai/whisper-base", align=False )["text"]
		
		lines = sentence_split(text, split_by=sampling_kwargs.get("split_text_by", "sentences"))
		if not lines:
			lines = [""]
			
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

		modality = self.modality( modality )
		# force AR+NAR
		if modality == "ar+nar":
			model_len = None
		# force NAR-len
		elif modality == "nar-len":
			model_ar = None

		if task == "stt":
			resp = self.encode_audio( references )
			lang = self.encode_lang( language )
			
			resp = to_device(resp, device=self.device, dtype=torch.int16)
			lang = to_device(lang, device=self.device, dtype=torch.uint8)

			with torch.autocast(self.device, dtype=dtype, enabled=amp):
				model = model_ar if model_ar is not None else model_nar
				if model is not None:
					phns_list = model(
						phns_list=None, proms_list=[resp], lang_list=[lang], resps_list=[resp], task_list=[task],
						disable_tqdm=not use_tqdm,
						use_lora=use_lora,
						**sampling_kwargs,
					)
				else:
					raise Exception("!")
				
				phns_list = [ cfg.tokenizer.decode( text ).replace("   ", "_").replace(" ", "").replace("_", " ") for text in phns_list ]

			return phns_list[0]
		elif task in ["phn", "un-phn"]:
			lang = self.encode_lang( language )
			lang = to_device(lang, device=self.device, dtype=torch.uint8)
			
			with torch.autocast(self.device, dtype=dtype, enabled=amp):
				model = model_ar if model_ar is not None else model_nar
				if task == "phn":
					phns_list = None
					text_list = [ self.encode_text( text, phonemize=False ).to(device=self.device, dtype=torch.int16) ]
					output_tokenizer = cfg.tokenizer
				else:
					phns_list = [ self.encode_text( text ).to(device=self.device, dtype=torch.int16) ]
					text_list = None
					output_tokenizer = cfg.text_tokenizer

				if model is not None:
					phns_list = model(
						phns_list=phns_list, text_list=text_list, lang_list=[lang], task_list=[task],
						disable_tqdm=not use_tqdm,
						use_lora=use_lora,
						**sampling_kwargs,
					)
				else:
					raise Exception("!")
				
				phns_list = [ output_tokenizer.decode( text ).replace("   ", "_").replace(" ", "").replace("_", " ") for text in phns_list ]

			return phns_list[0]


		# stuff for rolling context
		prefix_context = None
		prefix_contexts = []
		context_history = sampling_kwargs.get("context_history", 0)

		auto_lang = not language or language == "auto"
		auto_text_lang = not text_language or text_language == "auto"
		
		vc_utterance = self.encode_audio( voice_convert, trim_length=0 ) if voice_convert else None
		prom = self.encode_audio( references, trim_length=input_prompt_length ) if references else None
		lang = self.encode_lang( language )
		
		if task in ["ns", "sr"]:
			prom = [
				task,
				prom
			]
		
		prom = to_device(prom, device=self.device, dtype=torch.int16)
		lang = to_device(lang, device=self.device, dtype=torch.uint8)
		
		for line in lines:
			if out_path is None:
				output_dir = Path("./data/results/")
				if not output_dir.exists():
					output_dir.mkdir(parents=True, exist_ok=True)
				out_path = output_dir / f"{time.time()}.wav"

			deduced_language = g2p.detect_language( line ) if auto_lang or auto_text_lang else language

			if auto_lang:
				language = deduced_language

			if auto_text_lang:
				text_language = deduced_language

			phns = self.encode_text( line, language=text_language, phonemize=phonemize )
			phns = to_device(phns, device=self.device, dtype=torch.uint8 if len(self.symmap) < 256 else torch.int16)

			with torch.autocast(self.device, dtype=dtype, enabled=amp):
				input_kwargs = dict(
					phns_list=[phns] if phonemize else None,
					text_list=[phns] if not phonemize else None,
					proms_list=[prom],
					lang_list=[lang],
					disable_tqdm=not use_tqdm,
					use_lora=use_lora,
				)
				if model_len is not None:
					# skip calculating len_list if possible
					if task in ["ns", "sr"]:
						len_list = [ prom[1].shape[0] ]
					elif vc_utterance is not None:
						len_list = [ vc_utterance.shape[0] ]
					else:					
						len_list = model_len( **input_kwargs, task_list=["len"], **{"max_duration": 5} ) # "max_duration" is max tokens

						# clamp
						len_list = [ max( l, 1 * cfg.dataset.frames_per_second ) for l in len_list ]

						# add an additional X seconds
						len_list = [ int(l * duration_padding) for l in len_list ]

					kwargs = {}
					if prefix_context is not None:
						kwargs["prefix_context"] = prefix_context
					if vc_utterance is not None:
						kwargs["vc_list"] = [ vc_utterance ]

					resps_list = model_nar( **input_kwargs, len_list=len_list, task_list=["tts"],
						**(sampling_kwargs | kwargs),
					)
				elif model_ar is not None:
					kwargs = {}
					if prefix_context is not None:
						kwargs["prefix_context"] = prefix_context

					resps_list = model_ar(
						**input_kwargs, task_list=["tts"],
						**(sampling_kwargs | kwargs),
					)

					resps_list = model_nar(
						**input_kwargs, resps_list=resps_list, task_list=["tts"],
						**sampling_kwargs,
					)
				else:
					raise Exception("!")
				"""
				len_list = [ 3 * cfg.dataset.frames_per_second ]
				resps_list = model_nar( **input_kwargs, len_list=len_list, task_list=["tts"],
					**(sampling_kwargs),
				)
				"""

			# to-do: care about batching later
			resps = resps_list[0]
			
			# store current context to use as the initial input for later
			if context_history > 0:
				# add to history
				prefix_contexts.append(( phns, resps, resps.shape[0] ))
				# then generate the prefix based on how much history to provide
				prefix_context = (
					[ torch.concat( [ x[0] for x in prefix_contexts[-context_history:] ] ) ],
					[ torch.concat( [ x[1] for x in prefix_contexts[-context_history:] ] ) ],
					[ sum([ x[2] for x in prefix_contexts[-context_history:] ]) ]
				)

			# write to file
			wav, sr = qnt.decode_to_file(resps, out_path, device=self.device)
			# add utterances
			wavs.append(wav)

			if play:
				sd.play(wav.cpu().numpy()[0], sr)
				sd.wait()


		# combine all utterances
		return (torch.concat(wavs, dim=-1), sr)
		
