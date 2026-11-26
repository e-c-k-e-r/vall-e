# NVIDIA's 44.1 kHz audio codec (nvidia/audio-codec-44khz: a HiFi-GAN encoder, grouped FSQ, a HiFi-GAN decoder) for
# inference, in plain PyTorch: the modules of NeMo's AudioCodecModel that encoding and decoding use, under the same names
# (so the checkpoint's state dict loads as it is), read straight from the released .nemo archive (its model_config.yaml
# and model_weights.ckpt) without the nemo-toolkit package, whose releases stopped importing on a plain install (3.0
# moved its core dependencies into extras and refuses to build the codec's class without its whole TTS collection).
# The weight norm is folded into plain weights at load (inference only); the discriminator and losses aren't built.
#
#   codec = AudioCodecModel.from_pretrained("nvidia/audio-codec-44khz").to(device).eval()
#   tokens, tokens_len = codec.encode(audio=wav, audio_len=lengths)        # wav [B, samples] at 44.1 kHz -> [B, 8, frames]
#   audio, audio_len = codec.decode(tokens=tokens, tokens_len=tokens_len)  # -> [B, samples], 512 per frame
import io
import tarfile
from pathlib import Path

import torch
import torch.nn as nn

REPO, FILE = "nvidia/audio-codec-44khz", "audio-codec-44khz.nemo"


def mask_sequence_tensor(x, lengths):
	"""Zero the positions of each batch row past its length (the last dimension is time)."""
	if lengths is None:
		return x
	mask = torch.arange(x.shape[-1], device=x.device)[None, :] < lengths[:, None]
	return x * (mask[:, None, :] if x.dim() == 3 else mask).to(x.dtype)


def get_padding(kernel_size, dilation=1):
	return (kernel_size * dilation - dilation) // 2


def get_down_sample_padding(kernel_size, stride):
	return (kernel_size - stride + 1) // 2


def get_up_sample_padding(kernel_size, stride):
	return (kernel_size - stride + 1) // 2, (kernel_size - stride) % 2


def snake(x, alpha, eps=1e-9):
	"""x + sin(alpha x)^2 / alpha (arXiv 2006.08195)."""
	return x + (alpha + eps).reciprocal() * torch.sin(alpha * x).pow(2)


class Snake(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.alpha = nn.Parameter(torch.ones(1, channels, 1))

	def forward(self, x):
		return snake(x, self.alpha)


class HalfSnake(nn.Module):
	"""Snake on the first half of the channels, leaky ReLU on the rest."""
	def __init__(self, channels):
		super().__init__()
		self.snake_channels = channels // 2
		self.snake_act = Snake(self.snake_channels)
		self.lrelu = nn.LeakyReLU()

	def forward(self, x):
		return torch.cat([self.snake_act(x[:, : self.snake_channels]), self.lrelu(x[:, self.snake_channels :])], dim=1)


class ClampActivation(nn.Module):
	def forward(self, x):
		return x.clamp(-1.0, 1.0)


class CodecActivation(nn.Module):
	def __init__(self, activation="elu", channels=1):
		super().__init__()
		activation = activation.lower()
		if activation == "elu":
			self.activation = nn.ELU()
		elif activation == "lrelu":
			self.activation = nn.LeakyReLU()
		elif activation == "snake":
			self.activation = Snake(channels)
		elif activation == "half_snake":
			self.activation = HalfSnake(channels)
		else:
			raise ValueError(f"unknown activation {activation}")

	def forward(self, x):
		return self.activation(x)


class Conv1dNorm(nn.Module):
	"""A reflection-padded Conv1d (weight norm folded in at load)."""
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None):
		super().__init__()
		if not padding:
			padding = get_padding(kernel_size=kernel_size, dilation=dilation)
		self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, padding_mode="reflect")

	def remove_weight_norm(self):  # already plain weights
		pass

	def forward(self, inputs, input_len):
		return mask_sequence_tensor(self.conv(inputs), input_len)


class ConvTranspose1dNorm(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1):
		super().__init__()
		padding, output_padding = get_up_sample_padding(kernel_size, stride)
		self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)

	def remove_weight_norm(self):
		pass

	def forward(self, inputs, input_len):
		return mask_sequence_tensor(self.conv(inputs), input_len)


class ResidualBlock(nn.Module):
	def __init__(self, channels, filters, kernel_size=3, dilation=1, activation="lrelu"):
		super().__init__()
		self.input_activation = CodecActivation(activation=activation, channels=channels)
		self.skip_activation = CodecActivation(activation=activation, channels=filters)
		self.input_conv = Conv1dNorm(channels, filters, kernel_size=kernel_size, dilation=dilation)
		self.skip_conv = Conv1dNorm(filters, channels, kernel_size=kernel_size)

	def remove_weight_norm(self):
		pass

	def forward(self, inputs, input_len):
		res = self.skip_conv(self.skip_activation(self.input_conv(self.input_activation(inputs), input_len)), input_len)
		return inputs + res


class HiFiGANResBlock(nn.Module):
	"""Residual blocks over the dilations, in a chain."""
	def __init__(self, channels, kernel_size, dilations, activation):
		super().__init__()
		self.res_blocks = nn.ModuleList([ResidualBlock(channels, channels, kernel_size, d, activation) for d in dilations])

	def remove_weight_norm(self):
		pass

	def forward(self, inputs, input_len):
		out = inputs
		for block in self.res_blocks:
			out = block(out, input_len)
		return out


class HiFiGANResLayer(nn.Module):
	"""The mean of one HiFiGANResBlock per kernel size, each from the layer's input."""
	def __init__(self, channels, kernel_sizes, dilations, activation):
		super().__init__()
		self.res_blocks = nn.ModuleList([HiFiGANResBlock(channels, k, dilations, activation) for k in kernel_sizes])

	def remove_weight_norm(self):
		pass

	def forward(self, inputs, input_len):
		return sum(block(inputs, input_len) for block in self.res_blocks) / len(self.res_blocks)


class HiFiGANEncoder(nn.Module):
	def __init__(self, encoded_dim, down_sample_rates=(2, 2, 8, 8), base_channels=32, in_kernel_size=7, out_kernel_size=7,
			resblock_kernel_sizes=(3, 7, 11), resblock_dilation_sizes=(1, 3, 5), activation="lrelu"):
		super().__init__()
		self.down_sample_rates = down_sample_rates
		self.pre_conv = Conv1dNorm(1, base_channels, kernel_size=in_kernel_size)
		channels = base_channels
		self.activations, self.down_sample_conv_layers, self.res_layers = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
		for rate in down_sample_rates:
			self.res_layers.append(HiFiGANResLayer(channels, resblock_kernel_sizes, resblock_dilation_sizes, activation))
			self.activations.append(CodecActivation(activation, channels=channels))
			k = 2 * rate
			self.down_sample_conv_layers.append(Conv1dNorm(channels, 2 * channels, kernel_size=k, stride=rate, padding=get_down_sample_padding(k, rate)))
			channels *= 2
		self.post_activation = CodecActivation(activation, channels=channels)
		self.post_conv = Conv1dNorm(channels, encoded_dim, kernel_size=out_kernel_size)

	def remove_weight_norm(self):
		pass

	def forward(self, audio, audio_len):
		encoded_len = audio_len
		out = self.pre_conv(audio[:, None, :], encoded_len)
		for act, res_layer, down, rate in zip(self.activations, self.res_layers, self.down_sample_conv_layers, self.down_sample_rates):
			out = act(res_layer(out, encoded_len))
			encoded_len = encoded_len // rate
			out = down(out, encoded_len)
		return self.post_conv(self.post_activation(out), encoded_len), encoded_len


class HiFiGANDecoder(nn.Module):
	def __init__(self, input_dim, up_sample_rates=(8, 8, 2, 2), base_channels=512, in_kernel_size=7, out_kernel_size=3,
			resblock_kernel_sizes=(3, 7, 11), resblock_dilation_sizes=(1, 3, 5), activation="lrelu", output_activation="tanh"):
		super().__init__()
		self.up_sample_rates = up_sample_rates
		self.pre_conv = Conv1dNorm(input_dim, base_channels, kernel_size=in_kernel_size)
		channels = base_channels
		self.activations, self.up_sample_conv_layers, self.res_layers = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
		for rate in up_sample_rates:
			self.activations.append(CodecActivation(activation, channels=channels))
			self.up_sample_conv_layers.append(ConvTranspose1dNorm(channels, channels // 2, kernel_size=2 * rate, stride=rate))
			channels //= 2
			self.res_layers.append(HiFiGANResLayer(channels, resblock_kernel_sizes, resblock_dilation_sizes, activation))
		self.post_activation = CodecActivation(activation, channels=channels)
		self.post_conv = Conv1dNorm(channels, 1, kernel_size=out_kernel_size)
		if output_activation == "tanh":
			self.out_activation = nn.Tanh()
		elif output_activation == "clamp":
			self.out_activation = ClampActivation()
		else:
			raise ValueError(f"unknown output activation {output_activation}")

	def remove_weight_norm(self):
		pass

	def forward(self, inputs, input_len):
		audio_len = input_len
		out = self.pre_conv(inputs, audio_len)
		for act, res_layer, up, rate in zip(self.activations, self.res_layers, self.up_sample_conv_layers, self.up_sample_rates):
			audio_len = audio_len * rate
			out = res_layer(up(act(out), audio_len), audio_len)
		audio = self.out_activation(self.post_conv(self.post_activation(out), audio_len))
		return audio[:, 0], audio_len


class FiniteScalarQuantizer(nn.Module):
	"""FSQ (arXiv 2309.15505): each dimension compressed by tanh into its level range and rounded."""
	def __init__(self, num_levels, eps=1e-3):
		super().__init__()
		self.register_buffer("dim_base_index", torch.cumprod(torch.tensor([1] + list(num_levels[:-1])), dim=0, dtype=torch.int32)[None, :, None])
		self.register_buffer("num_levels", torch.tensor(num_levels, dtype=torch.int32)[None, :, None])
		self.eps = eps

	@property
	def dim(self):
		return self.num_levels.numel()

	def compress(self, inputs):
		output_scale = (self.num_levels - 1) / 2 * (1 - self.eps)
		output_offset = torch.where(self.num_levels % 2 == 0, 0.5, 0)
		input_shift = (output_offset / output_scale).tan()
		return output_scale * (inputs + input_shift).tanh() - output_offset

	def encode(self, inputs, input_len=None):
		"""[B, D, T] values -> [1, B, T] indices (one codebook)."""
		half = self.num_levels // 2
		codes = torch.round(self.compress(inputs)) / half                     # in [-1, 1], as NeMo's inputs_to_codes
		indices = torch.sum((half * codes + half) * self.dim_base_index, dim=1).to(torch.int32)
		return mask_sequence_tensor(indices, input_len)[None]

	def decode(self, indices, input_len=None):
		"""[1, B, T] indices -> [B, D, T] values in [-1, 1]."""
		nonneg = (indices[0][:, None, :] // self.dim_base_index) % self.num_levels
		return mask_sequence_tensor((nonneg - self.num_levels // 2) / (self.num_levels // 2), input_len)


class GroupFiniteScalarQuantizer(nn.Module):
	def __init__(self, num_groups, num_levels_per_group, **kwargs):
		super().__init__()
		self.num_groups = num_groups
		self.codebook_dim_per_group = len(num_levels_per_group)
		self.fsqs = nn.ModuleList([FiniteScalarQuantizer(num_levels_per_group, **kwargs) for _ in range(num_groups)])

	def encode(self, inputs, input_len):
		return torch.cat([fsq.encode(x, input_len) for x, fsq in zip(inputs.chunk(self.num_groups, dim=1), self.fsqs)], dim=0)

	def decode(self, indices, input_len):
		return torch.cat([fsq.decode(i, input_len) for i, fsq in zip(indices.chunk(self.num_groups, dim=0), self.fsqs)], dim=1)


_CLASSES = dict(HiFiGANEncoder=HiFiGANEncoder, HiFiGANDecoder=HiFiGANDecoder, GroupFiniteScalarQuantizer=GroupFiniteScalarQuantizer)


def _build(cfg):
	cfg = dict(cfg)
	name = cfg.pop("_target_").rsplit(".", 1)[-1]
	if name not in _CLASSES:
		raise ValueError(f"the codec's {name} isn't supported here")
	return _CLASSES[name](**cfg)


def _fold_weight_norm(state):
	"""torch.nn.utils.weight_norm's (weight_g, weight_v) pairs as plain weights: g * v / ||v||, the norm over every dimension
	but the first (its default dim=0, for convolutions and transposed ones alike)."""
	out = {}
	for k, v in state.items():
		if k.endswith(".weight_v"):
			base = k[: -len("_v")]
			out[base] = torch._weight_norm(v, state[base + "_g"], 0)   # what nn.utils.weight_norm computes on every forward
		elif not k.endswith(".weight_g"):
			out[k] = v
	return out


class AudioCodecModel(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.sample_rate = cfg["sample_rate"]
		self.samples_per_frame = cfg["samples_per_frame"]
		self.audio_encoder = _build(cfg["audio_encoder"])
		self.vector_quantizer = _build(cfg["vector_quantizer"])
		self.audio_decoder = _build(cfg["audio_decoder"])

	@classmethod
	def restore_from(cls, path):
		"""A .nemo archive: model_config.yaml and model_weights.ckpt (the encoder, quantizer and decoder are kept)."""
		import yaml
		with tarfile.open(path) as tar:
			names = {Path(m.name).name: m for m in tar.getmembers()}
			cfg = yaml.safe_load(tar.extractfile(names["model_config.yaml"]))
			state = torch.load(io.BytesIO(tar.extractfile(names["model_weights.ckpt"]).read()), map_location="cpu", weights_only=True)
		model = cls(cfg)
		keep = ("audio_encoder.", "vector_quantizer.", "audio_decoder.")
		model.load_state_dict(_fold_weight_norm({k: v for k, v in state.items() if k.startswith(keep)}))
		return model.eval()

	@classmethod
	def from_pretrained(cls, name=REPO):
		"""The codec from the Hugging Face hub (its cached copy when there is one, so it works offline)."""
		from huggingface_hub import hf_hub_download
		try:
			path = hf_hub_download(name, FILE, local_files_only=True)
		except Exception:
			path = hf_hub_download(name, FILE)
		return cls.restore_from(path)

	def pad_audio(self, audio, audio_len):
		"""Zero-pad the end so every row holds whole frames."""
		padded_len = self.samples_per_frame * torch.ceil(audio_len / self.samples_per_frame).int()
		return nn.functional.pad(audio, (0, int(padded_len.max()) - audio.shape[1])), padded_len

	@torch.no_grad()
	def encode(self, audio, audio_len):
		"""[B, samples] audio -> ([B, groups, frames] tokens, [B] frame counts)."""
		audio, audio_len = self.pad_audio(audio, audio_len)
		encoded, encoded_len = self.audio_encoder(audio, audio_len)
		return self.vector_quantizer.encode(encoded, encoded_len).permute(1, 0, 2), encoded_len

	@torch.no_grad()
	def dequantize(self, tokens, tokens_len):
		return self.vector_quantizer.decode(tokens.permute(1, 0, 2), tokens_len)

	@torch.no_grad()
	def decode_audio(self, inputs, input_len):
		return self.audio_decoder(inputs, input_len)

	@torch.no_grad()
	def decode(self, tokens, tokens_len):
		"""[B, groups, frames] tokens -> ([B, samples] audio, [B] sample counts)."""
		return self.decode_audio(self.dequantize(tokens, tokens_len), tokens_len)
