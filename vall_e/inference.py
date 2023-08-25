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
from .train import load_engines
from .data import get_phone_symmap, _load_quants

class TTS():
	def __init__( self, config=None, ar_ckpt=None, nar_ckpt=None, device="cuda" ):
		self.loading = True 
		self.device = device

		self.input_sample_rate = 24000
		self.output_sample_rate = 24000

		if config:
			cfg.load_yaml( config )
			cfg.dataset.use_hdf5 = False # could use cfg.load_hdf5(), but why would it ever need to be loaded for inferencing

		try:
			cfg.format()
		except Exception as e:
			pass
		
		self.symmap = None
		if ar_ckpt and nar_ckpt:
			self.ar_ckpt = ar_ckpt
			self.nar_ckpt = nar_ckpt

			models = get_models(cfg.models.get())
			for name, model in models.items():
				if name.startswith("ar"):
					self.ar = model
					state = torch.load(self.ar_ckpt)
					if "symmap" in state:
						self.symmap = state['symmap']
					if "module" in state:
						state = state['module']
					self.ar.load_state_dict(state)
					self.ar = self.ar.to(self.device, dtype=cfg.inference.dtype)
				elif name.startswith("nar"):
					self.nar = model
					state = torch.load(self.nar_ckpt)
					if "symmap" in state:
						self.symmap = state['symmap']
					if "module" in state:
						state = state['module']
					self.nar.load_state_dict(state)
					self.nar = self.nar.to(self.device, dtype=cfg.inference.dtype)
		else:
			self.load_models()

		if self.symmap is None:
			self.symmap = get_phone_symmap()

		self.ar.eval()
		self.nar.eval()

		self.loading = False 

	def load_models( self ):
		engines = load_engines()
		for name, engine in engines.items():
			if name[:2] == "ar":
				self.ar = engine.module.to(self.device, dtype=cfg.inference.dtype)
			elif name[:3] == "nar":
				self.nar = engine.module.to(self.device, dtype=cfg.inference.dtype)

	def encode_text( self, text, language="en" ):
		# already a tensor, return it
		if isinstance( text, Tensor ):
			return text

		content = g2p.encode(text, language=language)
		# ick
		try:
			phones = ["<s>"] + [ " " if not p else p for p in content ] + ["</s>"]
			return torch.tensor([*map(self.symmap.get, phones)])
		except Exception as e:
			pass
		phones = [ " " if not p else p for p in content ]
		return torch.tensor([ 1 ] + [*map(self.symmap.get, phones)] + [ 2 ])

	def encode_audio( self, paths, should_trim=True ):
		# already a tensor, return it
		if isinstance( paths, Tensor ):
			return paths

		# split string into paths
		if isinstance( paths, str ):
			paths = [ Path(p) for p in paths.split(";") ]

		# merge inputs
		res = torch.cat([qnt.encode_from_file(path)[0][:, :].t().to(torch.int16) for path in paths])
		
		if should_trim:
			res = trim( res, int( 75 * cfg.dataset.prompt_duration ) )
		
		return res

	@torch.inference_mode()
	def inference( self, text, references, max_ar_steps=6 * 75, ar_temp=1.0, nar_temp=1.0, out_path=None ):
		if out_path is None:
			out_path = f"./data/{text}.wav"

		prom = self.encode_audio( references )
		phns = self.encode_text( text )

		prom = to_device(prom, self.device).to(torch.int16)
		phns = to_device(phns, self.device).to(torch.uint8 if len(self.symmap) < 256 else torch.int16)

		resps_list = self.ar(text_list=[phns], proms_list=[prom], max_steps=max_ar_steps, sampling_temperature=ar_temp)
		resps_list = [r.unsqueeze(-1) for r in resps_list]
		resps_list = self.nar(text_list=[phns], proms_list=[prom], resps_list=resps_list, sampling_temperature=nar_temp)

		wav, sr = qnt.decode_to_file(resps_list[0], out_path)
		
		return (wav, sr)
		
