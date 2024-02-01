import torch
import torchaudio
import soundfile

from torch import Tensor
from einops import rearrange
from pathlib import Path

from .emb import g2p, qnt
from .emb.qnt import trim, trim_random
from .utils import to_device

from .config import cfg
from .models import get_models
from .engines import load_engines, deepspeed_available
from .data import get_phone_symmap, get_lang_symmap, _load_quants, _cleanup_phones

if deepspeed_available:
	import deepspeed

class TTS():
	def __init__( self, config=None, ar_ckpt=None, nar_ckpt=None, device=None, amp=None, dtype=None ):
		self.loading = True 
		
		self.input_sample_rate = 24000
		self.output_sample_rate = 24000

		if config:
			cfg.load_yaml( config )
			cfg.dataset.use_hdf5 = False # could use cfg.load_hdf5(), but why would it ever need to be loaded for inferencing

		try:
			cfg.format()
		except Exception as e:
			pass

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

		self.symmap = None

		def parse( name, model, state ):
			if "userdata" in state and 'symmap' in state['userdata']:
				self.symmap = state['userdata']['symmap']
			elif "symmap" in state:
				self.symmap = state['symmap']

			if "module" in state:
				state = state['module']
			
			model.load_state_dict(state)

			if cfg.inference.backend == "local" and deepspeed_available and cfg.trainer.deepspeed.inferencing:
				model = deepspeed.init_inference(model=model, mp_size=1, replace_with_kernel_inject=True, dtype=dtype if not amp else torch.float32).module

			return model

		if ar_ckpt and nar_ckpt:
			self.ar_ckpt = ar_ckpt
			self.nar_ckpt = nar_ckpt

			models = get_models(cfg.models.get(), training=False)

			for name, model in models.items():
				if name.startswith("ar"):
					state = torch.load(self.ar_ckpt)
					self.ar = parse( name, model, state )
				elif name.startswith("nar"):
					state = torch.load(self.nar_ckpt)
					self.nar = parse( name, model, state )
					
				if name.startswith("ar+nar"):
					self.nar = self.ar
		else:
			self.load_models()

		if self.dtype != torch.int8:
			self.ar = self.ar.to(self.device, dtype=self.dtype if not self.amp else torch.float32)
			self.nar = self.nar.to(self.device, dtype=self.dtype if not self.amp else torch.float32)

		self.ar.eval()
		self.nar.eval()

		if self.symmap is None:
			self.symmap = get_phone_symmap()

		self.loading = False 

	def load_models( self ):
		engines = load_engines(training=False)
		for name, engine in engines.items():
			if name.startswith("ar"):
				self.ar = engine.module
			elif name.startswith("nar"):
				self.nar = engine.module

			if name.startswith("ar+nar"):
				self.nar = self.ar

	def encode_text( self, text, language="en" ):
		# already a tensor, return it
		if isinstance( text, Tensor ):
			return text

		content = g2p.encode(text, language=language)
		content = _cleanup_phones( content )
		# ick
		try:
			phones = ["<s>"] + [ " " if not p else p for p in content ] + ["</s>"]
			return torch.tensor([*map(self.symmap.get, phones)])
		except Exception as e:
			pass
		phones = [ " " if not p else p for p in content ]
		return torch.tensor([ 1 ] + [*map(self.symmap.get, phones)] + [ 2 ])

	def encode_lang( self, language ):
		symmap = get_lang_symmap()
		id = 0
		if language in symmap:
			id = symmap[language]
		return torch.tensor([ id ])

	def encode_audio( self, paths, trim_length=0.0 ):
		# already a tensor, return it
		if isinstance( paths, Tensor ):
			return paths

		# split string into paths
		if isinstance( paths, str ):
			paths = [ Path(p) for p in paths.split(";") ]

		# merge inputs
		res = torch.cat([qnt.encode_from_file(path)[0][:, :].t().to(torch.int16) for path in paths])
		
		if trim_length:
			res = trim( res, int( 75 * trim_length ) )
		
		return res

	@torch.inference_mode()
	def inference(
		self,
		text,
		references,
		language="en",
		max_ar_steps=6 * 75,
		max_ar_context=-1,
		max_nar_levels=7,
		input_prompt_length=0.0,
		ar_temp=0.95,
		nar_temp=0.5,
		min_ar_temp=0.95,
		min_nar_temp=0.5,
		top_p=1.0,
		top_k=0,
		repetition_penalty=1.0,
		repetition_penalty_decay=0.0,
		length_penalty=0.0,
		beam_width=0,
		mirostat_tau=0,
		mirostat_eta=0.1,
		out_path=None
	):
		lines = text.split("\n")

		wavs = []
		sr = None

		for line in lines:
			if out_path is None:
				out_path = f"./data/{cfg.start_time}.wav"

			prom = self.encode_audio( references, trim_length=input_prompt_length )
			phns = self.encode_text( line, language=language )
			lang = self.encode_lang( language )

			prom = to_device(prom, self.device).to(torch.int16)
			phns = to_device(phns, self.device).to(torch.uint8 if len(self.symmap) < 256 else torch.int16)
			lang = to_device(lang, self.device).to(torch.uint8)

			with torch.autocast("cuda", dtype=self.dtype, enabled=self.amp):
				resps_list = self.ar(
					text_list=[phns], proms_list=[prom], lang_list=[lang], max_steps=max_ar_steps, max_resp_context=max_ar_context,
					sampling_temperature=ar_temp,
					sampling_min_temperature=min_ar_temp,
					sampling_top_p=top_p, sampling_top_k=top_k,
					sampling_repetition_penalty=repetition_penalty, sampling_repetition_penalty_decay=repetition_penalty_decay,
					sampling_length_penalty=length_penalty,
					sampling_beam_width=beam_width,
					sampling_mirostat_tau=mirostat_tau,
					sampling_mirostat_eta=mirostat_eta,
				)
				resps_list = [r.unsqueeze(-1) for r in resps_list]
				resps_list = self.nar(
					text_list=[phns], proms_list=[prom], lang_list=[lang], resps_list=resps_list,
					max_levels=max_nar_levels,
					sampling_temperature=nar_temp,
					sampling_min_temperature=min_nar_temp,
					sampling_top_p=top_p, sampling_top_k=top_k,
					sampling_repetition_penalty=repetition_penalty, sampling_repetition_penalty_decay=repetition_penalty_decay,
				)

			wav, sr = qnt.decode_to_file(resps_list[0], out_path, device=self.device)
			wavs.append(wav)
		
		return (torch.concat(wavs, dim=-1), sr)
		
