import torch
import torchaudio
import soundfile

from einops import rearrange

from .emb import g2p, qnt
from .utils import to_device

from .config import cfg
from .export import load_models
from .models import get_models
from .data import get_phone_symmap

class TTS():
	def __init__( self, config=None, ar_ckpt=None, nar_ckpt=None, device="cuda" ):
		self.loading = True 
		self.device = device

		self.input_sample_rate = 24000
		self.output_sample_rate = 24000
		
		if ar_ckpt and nar_ckpt:
			self.ar_ckpt = ar_ckpt
			self.nar_ckpt = nar_ckpt

			models = get_models(cfg.models.get())
			for name, model in models.items():
				if name.startswith("ar"):
					self.ar = model.to(self.device, dtype=torch.float32)
					self.ar.load_state_dict(torch.load(self.ar_ckpt)['module'])
				elif name.startswith("nar"):
					self.nar = model.to(self.device, dtype=torch.float32)
					self.nar.load_state_dict(torch.load(self.nar_ckpt)['module'])
		else:
			self.load_models( config )

		self.symmap = get_phone_symmap()
		self.ar.eval()
		self.nar.eval()

		self.loading = False 

	def load_models( self, config_path ):
		if config_path:
			cfg.load_yaml( config_path )

		print("Loading models...")
		models = load_models()
		print("Loaded models")
		for name in models:
			model = models[name]
			if name[:2] == "ar":
				self.ar = model.to(self.device, dtype=torch.float32)
				self.symmap = self.ar.phone_symmap
			elif name[:3] == "nar":
				self.nar = model.to(self.device, dtype=torch.float32)
			else:
				print("Unknown:", name)

	def encode_text( self, text, lang_marker="en" ):
		text = g2p.encode(text)
		phones = [f"<{lang_marker}>"] + [ " " if not p else p for p in text ] + [f"</{lang_marker}>"]
		mapped = [self.symmap[p] for p in phones if p in self.symmap]
		return torch.tensor( mapped )

	def encode_audio( self, path ):
		enc = qnt.encode_from_file( path )
		return enc[0].t().to(torch.int16)


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
		
