# WavLM (arXiv 2110.13900) for the speaker-similarity model (vall_e/emb/ecapa_tdnn.py): WavLM.py and modules.py are
# Microsoft's code (github.com/microsoft/unilm/tree/master/wavlm, MIT, headers kept) as s3prl ships it (Apache 2.0),
# copied unchanged; UpstreamExpert reproduces s3prl's upstream wrapper. This replaced torch.hub.load("s3prl/s3prl",
# "wavlm_large"): s3prl's hub imports all of its ~80 upstreams, several of which no longer import with torchaudio 2.9+,
# and it downloaded a 1.2 GB checkpoint whose weights the speaker model's own checkpoint replaces anyway.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .WavLM import WavLM, WavLMConfig

# WavLM Large (s3prl's wavlm_large checkpoint's "cfg")
WAVLM_LARGE = dict(extractor_mode="layer_norm", encoder_layers=24, encoder_embed_dim=1024, encoder_ffn_embed_dim=4096,
	encoder_attention_heads=16, activation_fn="gelu", dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, encoder_layerdrop=0.0,
	dropout_input=0.0, dropout_features=0.0, layer_norm_first=True, conv_feature_layers="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
	conv_bias=False, feature_grad_mult=1.0, mask_length=10, mask_prob=0.8, mask_selection="static", mask_other=0.0, no_mask_overlap=False,
	mask_min_space=1, mask_channel_length=10, mask_channel_prob=0.0, mask_channel_selection="static", mask_channel_other=0.0,
	no_mask_channel_overlap=False, mask_channel_min_space=1, conv_pos=128, conv_pos_groups=16, relative_position_embedding=True,
	num_buckets=320, max_distance=800, gru_rel_pos=True, normalize=True)

CONFIGS = dict(wavlm_large=WAVLM_LARGE)


class UpstreamExpert(nn.Module):
	"""s3prl's WavLM upstream: a list of 16 kHz waveforms -> {"hidden_states": the input of every transformer layer and
	the encoder's output, each [batch, frames, dim]}, its hooks' outputs, in order. The weights come from the caller (the
	speaker model's checkpoint holds them), so none are downloaded."""
	def __init__(self, name="wavlm_large"):
		super().__init__()
		self.cfg = WavLMConfig(CONFIGS[name])
		self.model = WavLM(self.cfg)
		self.model.feature_grad_mult = 0.0
		self.model.encoder.layerdrop = 0.0

	def forward(self, wavs):
		if self.cfg.normalize:
			wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]
		device = wavs[0].device
		lengths = torch.LongTensor([len(w) for w in wavs]).to(device)
		padding_mask = ~torch.lt(torch.arange(max(lengths)).unsqueeze(0).to(device), lengths.unsqueeze(1))
		hidden = []
		hooks = [layer.register_forward_hook(lambda m, inputs, output: hidden.append(inputs[0].transpose(0, 1))) for layer in self.model.encoder.layers]
		hooks.append(self.model.encoder.register_forward_hook(lambda m, inputs, output: hidden.append(output[0])))
		try:
			self.model.extract_features(pad_sequence(wavs, batch_first=True), padding_mask=padding_mask, mask=False)
		finally:
			for h in hooks:
				h.remove()
		return {"hidden_states": tuple(hidden), "last_hidden_state": hidden[-1]}
