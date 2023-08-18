import torch
import torchaudio
import soundfile

from einops import rearrange

from .emb import g2p, qnt
from .utils import to_device

from .config import cfg
from .models import get_models
from .train import load_engines
from .data import get_phone_symmap

import random

def trim( qnt, trim_length  ):
	length = qnt.shape[0]
	start = int(length * random.random())
	end = start + trim_length
	if end >= length:
		start = length - trim_length
		end = length
	return qnt[start:end]

class TTS():
	def __init__( self, config=None, ar_ckpt=None, nar_ckpt=None, device="cuda" ):
		self.loading = True 
		self.device = device

		self.input_sample_rate = 24000
		self.output_sample_rate = 24000

		if config:
			cfg.load_yaml( config )

		cfg.format()

		"""
		if cfg.trainer.load_state_dict:
			for model in cfg.models.get():
				path = cfg.ckpt_dir / model.full_name / "fp32.pth"
				if model.name.startswith("ar"):
					ar_ckpt = path
				if model.name.startswith("nar"):
					nar_ckpt = path
		"""
		
		if ar_ckpt and nar_ckpt:
			self.ar_ckpt = ar_ckpt
			self.nar_ckpt = nar_ckpt

			models = get_models(cfg.models.get())
			for name, model in models.items():
				if name.startswith("ar"):
					self.ar = model.to(self.device, dtype=torch.float32)
					state = torch.load(self.ar_ckpt)
					if "module" in state:
						state = state['module']
					self.ar.load_state_dict(state)
				elif name.startswith("nar"):
					self.nar = model.to(self.device, dtype=torch.float32)
					state = torch.load(self.nar_ckpt)
					if "module" in state:
						state = state['module']
					self.nar.load_state_dict(state)
		else:
			self.load_models()

		self.symmap = get_phone_symmap()
		self.ar.eval()
		self.nar.eval()

		self.loading = False 

	def load_models( self ):
		engines = load_engines()
		for name, engine in engines.items():
			if name[:2] == "ar":
				self.ar = engine.module.to(self.device)
			elif name[:3] == "nar":
				self.nar = engine.module.to(self.device)

	def encode_text( self, text, lang_marker="en" ):
		text = g2p.encode(text)
		phones = [f"<{lang_marker}>"] + [ " " if not p else p for p in text ] + [f"</{lang_marker}>"]
		mapped = [self.symmap[p] for p in phones if p in self.symmap]
		return torch.tensor( mapped )

	def encode_audio( self, path ):
		enc = qnt.encode_from_file( path )
		res = enc[0].t().to(torch.int16)
		if trim:
			res = trim( res, int( 75 * cfg.dataset.duration_range[1] ) )
		return res


	def inference( self, text, reference, mode="both", max_ar_steps=6 * 75, ar_temp=1.0, nar_temp=1.0, out_path="./.tmp.wav" ):
		prom = self.encode_audio( reference )
		phns = self.encode_text( text )

		prom = to_device(prom, self.device).to(torch.int16)
		phns = to_device(phns, self.device).to(torch.uint8 if len(self.symmap) < 256 else torch.int16)

		resps_list = self.ar(text_list=[phns], proms_list=[prom], max_steps=max_ar_steps, sampling_temperature=ar_temp)
		resps_list = [r.unsqueeze(-1) for r in resps_list]
		resps_list = self.nar(text_list=[phns], proms_list=[prom], resps_list=resps_list, sampling_temperature=nar_temp)

		wav, sr = qnt.decode_to_file(resps_list[0], out_path)
		
		return (wav, sr)
		
