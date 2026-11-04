# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# modified grossly to avoid additional dependencies

CONSTANT = 1e-5

import librosa
import itertools
import random

from math import ceil
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio

from einops import rearrange

from nemo.core import ModelPT
from nemo.core.classes.module import NeuralModule
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.neural_type import NeuralType

from nemo.collections.common.parts.utils import ClampActivation, HalfSnake, Snake, mask_sequence_tensor

from nemo.utils import model_utils
from nemo.utils.decorators import experimental

from nemo.core.neural_types.elements import (
    AudioSignal,
    EncodedRepresentation,
    Index,
    LengthsType,
    MelSpectrogramType,
    VoidType,
    TokenIndex,
)

from nemo.core.classes import Loss

def instantiate( cfg ):
    cls = None
    cfg = dict(cfg)
    target = cfg.pop("_target_")

    if target == "nemo.collections.tts.modules.audio_codec_modules.HiFiGANEncoder":
        cls = HiFiGANEncoder
    elif target == "nemo.collections.tts.modules.audio_codec_modules.GroupFiniteScalarQuantizer":
        cls = GroupFiniteScalarQuantizer
    elif target == "nemo.collections.tts.modules.audio_codec_modules.HiFiGANDecoder":
        cls = HiFiGANDecoder
    elif target == "nemo.collections.tts.modules.audio_codec_modules.Discriminator":
        cls = Discriminator
        # cheat here
        cfg['discriminators'] = [ instantiate( c ) for c in cfg['discriminators'] ]
        # {'discriminators': [{'_target_': 'nemo.collections.tts.modules.audio_codec_modules.MultiPeriodDiscriminator'}, {'_target_': 'nemo.collections.tts.modules.audio_codec_modules.MultiResolutionDiscriminatorSTFT', 'resolutions': [[512, 128, 512], [1024, 256, 1024], [2048, 512, 2048]], 'stft_bands': [[0.0, 0.1], [0.1, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]]}]}
    elif target == "nemo.collections.tts.modules.audio_codec_modules.MultiPeriodDiscriminator":
        cls = MultiPeriodDiscriminator
    elif target == "nemo.collections.tts.modules.audio_codec_modules.MultiResolutionDiscriminatorSTFT":
        cls = MultiResolutionDiscriminatorSTFT
    elif target == "nemo.collections.tts.losses.audio_codec_loss.GeneratorSquaredLoss":
        cls = GeneratorSquaredLoss
    elif target == "nemo.collections.tts.losses.audio_codec_loss.DiscriminatorSquaredLoss":
        cls = DiscriminatorSquaredLoss
    else:
        print( target, cfg )
        raise Exception("!")

    return cls( **cfg )

class GaussianDropout(torch.nn.Module):
    """
    Gaussian dropout using multiplicative gaussian noise.

    https://keras.io/api/layers/regularization_layers/gaussian_dropout/

    Can be an effective alternative bottleneck to VAE or VQ:

    https://www.deepmind.com/publications/gaussian-dropout-as-an-information-bottleneck-layer

    Unlike some other implementations, this takes the standard deviation of the noise as input
    instead of the 'rate' typically defined as: stdev = sqrt(rate / (1 - rate))
    """

    def __init__(self, stdev=1.0):
        super(GaussianDropout, self).__init__()
        self.stdev = stdev

    def forward(self, inputs):
        if not self.training:
            return inputs

        noise = torch.normal(mean=1.0, std=self.stdev, size=inputs.shape, device=inputs.device)
        out = noise * inputs
        return out

def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def get_padding_2d(kernel_size: Tuple[int, int], dilation: Tuple[int, int]) -> Tuple[int, int]:
    paddings = (get_padding(kernel_size[0], dilation[0]), get_padding(kernel_size[1], dilation[1]))
    return paddings


def get_down_sample_padding(kernel_size: int, stride: int) -> int:
    return (kernel_size - stride + 1) // 2


def get_up_sample_padding(kernel_size: int, stride: int) -> Tuple[int, int]:
    output_padding = (kernel_size - stride) % 2
    padding = (kernel_size - stride + 1) // 2
    return padding, output_padding


class SSLModel(NeuralModule):
    def __init__(self, slm_model_name):
        super().__init__()
        self.ssl_model = AutoModel.from_pretrained(slm_model_name)

    def forward(self, *args, **kwargs):
        return self.ssl_model(*args, **kwargs)


class SLMDiscriminator(NeuralModule):
    """SLM Discriminator, as described in both the StyleTTS2 and Low Frame-Rate Speech Codec papers.

    Args:
        slm_model_name: Hugging Face Speech Language Models name.
        slm_sr: Speech Language Models input sampling rate.
        input_sr: Audio input sampling rate.
        slm_hidden: Speech Language Model hidden dim.
        slm_layers: Speech Language Model number of layers.
        initial_channel: discriminative head number of channels.
        use_spectral_norm: If True uses spectral normalization otherwise uses weight norm.

    """

    def __init__(
        self,
        slm_model_name="microsoft/wavlm-base-plus",
        slm_sr=16000,
        input_sr=22050,
        slm_hidden=768,
        slm_layers=13,
        initial_channel=64,
        use_spectral_norm=False,
    ):
        super().__init__()

        self.slm_model = SSLModel(slm_model_name)

        # Freeze slm model
        self.slm_model.freeze()

        self.resample = torchaudio.transforms.Resample(input_sr, slm_sr)

        norm_f = torch.nn.utils.weight_norm if use_spectral_norm == False else torch.nn.utils.spectral_norm
        self.pre = norm_f(nn.Conv1d(slm_hidden * slm_layers, initial_channel, 1, 1, padding=0))

        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(initial_channel, initial_channel * 2, kernel_size=5, padding=2)),
                norm_f(nn.Conv1d(initial_channel * 2, initial_channel * 4, kernel_size=5, padding=2)),
                norm_f(nn.Conv1d(initial_channel * 4, initial_channel * 4, 5, 1, padding=2)),
            ]
        )

        self.conv_post = norm_f(nn.Conv1d(initial_channel * 4, 1, 3, 1, padding=1))

    def _forward(self, x):
        x = self.slm_model(input_values=self.resample(x), output_hidden_states=True).hidden_states
        x = torch.stack(x, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

        x = self.pre(x)
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x.unsqueeze(-1))

        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    @property
    def input_types(self):
        return {
            "audio_real": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_gen": NeuralType(('B', 'T_audio'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "scores_real": [NeuralType(('B', 'C', 'T_out'), VoidType())],
            "scores_gen": [NeuralType(('B', 'C', 'T_out'), VoidType())],
            "fmaps_real": [[NeuralType(('B', 'D', 'T_layer', 'C'), VoidType())]],
            "fmaps_gen": [[NeuralType(('B', 'D', 'T_layer', 'C'), VoidType())]],
        }

    @typecheck()
    def forward(self, audio_real, audio_gen):

        y_d_r, fmap_r = self._forward(audio_real)
        y_d_g, fmap_g = self._forward(audio_gen)

        return [y_d_r.unsqueeze(1)], [y_d_g.unsqueeze(1)], [fmap_r], [fmap_g]


class CodecActivation(nn.Module):
    """
    Choose between activation based on the input parameter.

    Args:
        activation: Name of activation to use. Valid options are "elu" (default), "lrelu", and "snake".
        channels: Input dimension.
    """

    def __init__(self, activation: str = "elu", channels: int = 1):
        super().__init__()
        activation = activation.lower()
        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "lrelu":
            self.activation = torch.nn.LeakyReLU()
        elif activation == "snake":
            self.activation = Snake(channels)
        elif activation == "half_snake":
            self.activation = HalfSnake(channels)
        else:
            raise ValueError(f"Unknown activation {activation}")

    def forward(self, x):
        return self.activation(x)


class Conv1dNorm(NeuralModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        padding: Optional[int] = None,
    ):
        super().__init__()
        if not padding:
            padding = get_padding(kernel_size=kernel_size, dilation=dilation)
        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            padding_mode="reflect",
        )
        self.conv = nn.utils.weight_norm(conv)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'C', 'T'), VoidType()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'C', 'T'), VoidType()),
        }

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)

    @typecheck()
    def forward(self, inputs, input_len):
        out = self.conv(inputs)
        out = mask_sequence_tensor(out, input_len)
        return out


class ConvTranspose1dNorm(NeuralModule):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        padding, output_padding = get_up_sample_padding(kernel_size, stride)
        conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            padding_mode="zeros",
        )
        self.conv = nn.utils.weight_norm(conv)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'C', 'T'), VoidType()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'C', 'T'), VoidType()),
        }

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)

    @typecheck()
    def forward(self, inputs, input_len):
        out = self.conv(inputs)
        out = mask_sequence_tensor(out, input_len)
        return out


class Conv2dNorm(NeuralModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
    ):
        super().__init__()
        assert len(kernel_size) == len(dilation)
        padding = get_padding_2d(kernel_size, dilation)
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            padding_mode="reflect",
        )
        self.conv = nn.utils.weight_norm(conv)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'C', 'H', 'T'), VoidType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'C', 'H', 'T'), VoidType()),
        }

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)

    @typecheck()
    def forward(self, inputs):
        return self.conv(inputs)


class PeriodDiscriminator(NeuralModule):
    """
    Period discriminator introduced in HiFi-GAN https://arxiv.org/abs/2010.05646 which attempts to
    discriminate phase information by looking at equally spaced audio samples.

    Args:
        period: Spacing between audio sample inputs.
        lrelu_slope: Slope to use for activation. Leaky relu with slope of 0.1 or 0.2 is recommended for the
           stability of the feature matching loss.
    """

    def __init__(self, period, lrelu_slope=0.1):
        super().__init__()
        self.period = period
        self.activation = nn.LeakyReLU(lrelu_slope)
        self.conv_layers = nn.ModuleList(
            [
                Conv2dNorm(1, 32, kernel_size=(5, 1), stride=(3, 1)),
                Conv2dNorm(32, 128, kernel_size=(5, 1), stride=(3, 1)),
                Conv2dNorm(128, 512, kernel_size=(5, 1), stride=(3, 1)),
                Conv2dNorm(512, 1024, kernel_size=(5, 1), stride=(3, 1)),
                Conv2dNorm(1024, 1024, kernel_size=(5, 1), stride=(1, 1)),
            ]
        )
        self.conv_post = Conv2dNorm(1024, 1, kernel_size=(3, 1))

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "score": NeuralType(('B', 'C', 'T_out'), VoidType()),
            "fmap": [NeuralType(('B', 'D', 'T_layer', 'C'), VoidType())],
        }

    @typecheck()
    def forward(self, audio):

        batch_size, time = audio.shape
        out = rearrange(audio, 'B T -> B 1 T')
        # Pad audio so that it is divisible by the period
        if time % self.period != 0:
            n_pad = self.period - (time % self.period)
            out = F.pad(out, (0, n_pad), "reflect")
            time = time + n_pad
        # [batch, 1, (time / period), period]
        out = out.view(batch_size, 1, time // self.period, self.period)

        fmap = []
        for conv in self.conv_layers:
            # [batch, filters, (time / period / stride), period]
            out = conv(inputs=out)
            out = self.activation(out)
            fmap.append(out)
        # [batch, 1, (time / period / strides), period]
        score = self.conv_post(inputs=out)
        fmap.append(score)
        score = rearrange(score, "B 1 T C -> B C T")

        return score, fmap


class MultiPeriodDiscriminator(NeuralModule):
    """
    Wrapper class to aggregate results of multiple period discriminators.

    The periods are expected to be increasing prime numbers in order to maximize coverage and minimize overlap
    """

    def __init__(self, periods: Iterable[int] = (2, 3, 5, 7, 11), lrelu_slope=0.1):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [PeriodDiscriminator(period=period, lrelu_slope=lrelu_slope) for period in periods]
        )

    @property
    def input_types(self):
        return {
            "audio_real": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_gen": NeuralType(('B', 'T_audio'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "scores_real": [NeuralType(('B', 'C', 'T_out'), VoidType())],
            "scores_gen": [NeuralType(('B', 'C', 'T_out'), VoidType())],
            "fmaps_real": [[NeuralType(('B', 'D', 'T_layer', 'C'), VoidType())]],
            "fmaps_gen": [[NeuralType(('B', 'D', 'T_layer', 'C'), VoidType())]],
        }

    @typecheck()
    def forward(self, audio_real, audio_gen):
        scores_real = []
        scores_gen = []
        fmaps_real = []
        fmaps_gen = []
        for discriminator in self.discriminators:
            score_real, fmap_real = discriminator(audio=audio_real)
            score_gen, fmap_gen = discriminator(audio=audio_gen)
            scores_real.append(score_real)
            fmaps_real.append(fmap_real)
            scores_gen.append(score_gen)
            fmaps_gen.append(fmap_gen)

        return scores_real, scores_gen, fmaps_real, fmaps_gen


class DiscriminatorSTFT(NeuralModule):
    """
    Discriminator network from EnCodec for Complex STFT input, but without dilations.

    Args:
        filters: number of filters to use in Conv2d layers
        lrelu_slope: Slope to use for activations. Leaky relu with slope of 0.1 or 0.2 is recommended for the
           stability of the feature matching loss
    """

    def __init__(self, filters: int = 32, lrelu_slope: float = 0.1):
        super().__init__()

        self.activation = nn.LeakyReLU(lrelu_slope)
        self.conv_layers = nn.ModuleList(
            [
                Conv2dNorm(2, filters, kernel_size=(3, 9)),
                Conv2dNorm(filters, filters, kernel_size=(3, 9), stride=(1, 2)),
                Conv2dNorm(filters, filters, kernel_size=(3, 9), stride=(1, 2)),
                Conv2dNorm(filters, filters, kernel_size=(3, 9), stride=(1, 2)),
                Conv2dNorm(filters, filters, kernel_size=(3, 3)),
            ]
        )
        self.conv_post = Conv2dNorm(filters, 1, kernel_size=(3, 3))

    @property
    def input_types(self):
        return {
            "spec": NeuralType(('B', 'C', 'T_spec', 'D'), VoidType()),
        }

    @property
    def output_types(self):
        return {
            "scores": NeuralType(('B', 'C', 'T_spec'), VoidType()),
            "fmap": [NeuralType(('B', 'D', 'T_spec', 'C'), VoidType())],
        }

    @typecheck()
    def forward(self, spec):
        fmap = []

        # [batch, 2, T_spec, fft]
        out = spec
        for conv in self.conv_layers:
            # [batch, filters, T_spec, fft // strides]
            out = conv(inputs=out)
            out = self.activation(out)
            fmap.append(out)
        # [batch, 1, T_spec, fft // 8]
        scores = self.conv_post(inputs=out)
        fmap.append(scores)
        scores = rearrange(scores, "B 1 T C -> B C T")

        return scores, fmap


class MultiBandDiscriminatorSTFT(NeuralModule):
    """
    Multi-band STFT discriminator proposed in DAC (https://arxiv.org/abs/2306.06546).

    Computes the complex STFT for a given resolution and splits it into sub-bands,
    which are given to separate discriminator networks.

    Args:
        resolution: STFT resolution, provided as a tuple of 3 integers ordered (num_fft, hop_length, window_length)
        stft_bands: List of tuples, with each tuple having 2 float values (band_start, band_end).
            The floats are in the range [0, 1] representing the fraction of all stft bands.
            For example for n_fft=1024, the stft output has 513 dimensions.
            For band input [(0, 0.25), (0.25, 1.0)] it would use stft dimensions [0 through 127] and [128 through 512].
    """

    def __init__(self, resolution: Tuple[int], stft_bands: Iterable[Tuple[int]]):
        super().__init__()

        self.n_fft, self.hop_length, self.win_length = resolution
        self.register_buffer("window", torch.hann_window(self.win_length, periodic=False))
        self.discriminators = nn.ModuleList([DiscriminatorSTFT() for _ in stft_bands])
        n_stft = self.n_fft // 2 + 1
        self.stft_bands = [(int(band[0] * n_stft), int(band[1] * n_stft)) for band in stft_bands]

    def compute_stft(self, audio):
        # [B, fft, T_spec]
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=True,
            center=True,
            return_complex=True,
        )
        fft = rearrange(fft, "B fft T -> B T fft")
        # [batch, 2, T_spec, fft]
        out = torch.stack([fft.real, fft.imag], dim=1)
        return out

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "scores_list": [NeuralType(('B', 'C', 'T_spec'), VoidType())],
            "fmaps_list": [[NeuralType(('B', 'D', 'T_spec', 'C'), VoidType())]],
        }

    @typecheck()
    def forward(self, audio):
        scores_list = []
        fmap_list = []
        spec = self.compute_stft(audio)
        for band, disc in zip(self.stft_bands, self.discriminators):
            spec_band = spec[:, :, :, band[0] : band[1]]
            score, fmap = disc(spec=spec_band)
            scores_list.append(score)
            fmap_list.append(fmap)

        return scores_list, fmap_list


class MultiResolutionDiscriminatorSTFT(NeuralModule):
    """
    Multi-resolution discriminator which creates a multi-band discriminator for each input resolution.

    Args:
        resolutions: List of STFT resolutions, each resolution provided as a tuple of 3 integers ordered
            (num_fft, hop_length, window_length)
        stft_bands: List of tuples, with each tuple having 2 float values (band_start, band_end).
            The floats are in the range [0, 1] representing the fraction of all stft bands.
            For example for n_fft=1024, the stft output has 513 dimensions.
            For band input [(0, 0.25), (0.25, 1.0)] it would use stft dimensions [0 through 127] and [128 through 512].
    """

    def __init__(self, resolutions: Iterable[Tuple[int]], stft_bands: Iterable[Tuple[int]]):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [MultiBandDiscriminatorSTFT(resolution=resolution, stft_bands=stft_bands) for resolution in resolutions]
        )

    @property
    def input_types(self):
        return {
            "audio_real": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_gen": NeuralType(('B', 'T_audio'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "scores_real": [NeuralType(('B', 'C', 'T_spec'), VoidType())],
            "scores_gen": [NeuralType(('B', 'C', 'T_spec'), VoidType())],
            "fmaps_real": [[NeuralType(('B', 'D', 'T_spec', 'C'), VoidType())]],
            "fmaps_gen": [[NeuralType(('B', 'D', 'T_spec', 'C'), VoidType())]],
        }

    @typecheck()
    def forward(self, audio_real, audio_gen):
        scores_real = []
        scores_gen = []
        fmaps_real = []
        fmaps_gen = []

        for disc in self.discriminators:
            score_real_i, fmap_real_i = disc(audio=audio_real)
            scores_real = scores_real + score_real_i
            fmaps_real = fmaps_real + fmap_real_i

            score_gen_i, fmap_gen_i = disc(audio=audio_gen)
            scores_gen = scores_gen + score_gen_i
            fmaps_gen = fmaps_gen + fmap_gen_i

        return scores_real, scores_gen, fmaps_real, fmaps_gen


class Discriminator(NeuralModule):
    """
    Wrapper class which takes a list of discriminators and aggregates the results across them.
    """

    def __init__(self, discriminators: Iterable[NeuralModule]):
        super().__init__()
        self.discriminators = nn.ModuleList(discriminators)

    @property
    def input_types(self):
        return {
            "audio_real": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_gen": NeuralType(('B', 'T_audio'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "scores_real": [NeuralType(('B', 'C', 'T_out'), VoidType())],
            "scores_gen": [NeuralType(('B', 'C', 'T_out'), VoidType())],
            "fmaps_real": [[NeuralType(('B', 'D', 'T_layer', 'C'), VoidType())]],
            "fmaps_gen": [[NeuralType(('B', 'D', 'T_layer', 'C'), VoidType())]],
        }

    @typecheck()
    def forward(self, audio_real, audio_gen):
        scores_real = []
        scores_gen = []
        fmaps_real = []
        fmaps_gen = []
        for discriminator in self.discriminators:
            score_real, score_gen, fmap_real, fmap_gen = discriminator(audio_real=audio_real, audio_gen=audio_gen)
            scores_real += score_real
            fmaps_real += fmap_real
            scores_gen += score_gen
            fmaps_gen += fmap_gen

        return scores_real, scores_gen, fmaps_real, fmaps_gen


class VectorQuantizerBase(NeuralModule, ABC):
    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "dequantized": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "indices": NeuralType(('D', 'B', 'T'), Index()),
        }

    @typecheck()
    @abstractmethod
    def forward(self, inputs: torch.Tensor, input_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"indices": NeuralType(('D', 'B', 'T'), Index())},
    )
    @abstractmethod
    def encode(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        pass

    @typecheck(
        input_types={
            "indices": NeuralType(('D', 'B', 'T'), Index()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "dequantized": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
        },
    )
    @abstractmethod
    def decode(self, indices: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        pass


class FiniteScalarQuantizer(VectorQuantizerBase):
    """This quantizer is based on the Finite Scalar Quantization (FSQ) method.
    It quantizes each element of the input vector independently into a number of levels.

    Args:
        num_levels: number of levels for each dimension/element of the input vector
        eps: small regularization constant for scaling

    References:
        Mentzer et al., Finite Scalar Quantization: VQ-VAE Made Simple (https://arxiv.org/abs/2309.15505v1)
    """

    def __init__(self, num_levels: List[int], eps: float = 1e-3):
        super().__init__()

        # index base per dimension of the input vector
        # this is used to convert between per-dimension indices and a codebook token index
        dim_base_index = torch.cumprod(torch.tensor([1] + num_levels[:-1]), dim=0, dtype=torch.int32)
        dim_base_index = rearrange(dim_base_index, 'D -> 1 D 1')
        self.register_buffer('dim_base_index', dim_base_index)

        # Register the number of levels for each dimension
        num_levels = torch.tensor(num_levels, dtype=torch.int32)
        num_levels = rearrange(num_levels, 'D -> 1 D 1')
        self.register_buffer('num_levels', num_levels)

        # Regularization
        self.eps = eps

    @property
    def codebook_size(self):
        """Returns the size of the corresponding codebook."""
        return self.num_levels.prod().item()

    @property
    def dim(self):
        """Returns the dimension of the input vector."""
        return self.num_levels.numel()

    @property
    def codebook_dim(self):
        """Returns the dimension of the input vector.
        Keeping for compatiblitiy with the original RVQ implementation.
        """
        return self.dim

    @property
    def codes(self):
        """Returns the codebooks entries.

        Note that the codebook entries are implicitly defined by the number of levels.
        """
        indices = torch.arange(self.codebook_size)
        # [D, B, T]
        indices = rearrange(indices, 'B -> 1 B 1')
        # [B, D, T]
        codes = self.decode(indices=indices, input_len=None)
        # Remove the time dimension
        codes = codes.squeeze(-1)
        return codes

    @property
    def codebook(self):
        """Returns the codebooks entries.
        See self.codes for more details.
        """
        return self.codes

    @staticmethod
    def round(inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        """Round the input tensor to nearest integer
        and use a straight-through estimator for the gradient.
        """
        inputs_rounded = torch.round(inputs)
        return inputs + (inputs_rounded - inputs).detach()

    def compress(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        """Apply compression to the input, to limit to values."""
        output_scale = (self.num_levels - 1) / 2
        # scale down a bit to avoid rounding issues
        output_scale = output_scale * (1 - self.eps)
        # offset for even number of levels
        output_offset = torch.where(self.num_levels % 2 == 0, 0.5, 0)
        # shift for even number of levels
        input_shift = (output_offset / output_scale).tan()
        # compressed output
        output = output_scale * (inputs + input_shift).tanh() - output_offset
        return output

    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"codes": NeuralType(('B', 'D', 'T'), Index())},
    )
    def inputs_to_codes(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        # apply compression
        compressed = self.compress(inputs=inputs, input_len=input_len)
        # apply rounding to nearest integer
        codes = self.round(inputs=compressed, input_len=input_len)
        # normalize to [-1, 1]
        scale = self.num_levels // 2
        codes = codes / scale
        return codes

    def codes_to_nonnegative(self, codes: torch.Tensor) -> torch.Tensor:
        """Convert values centered arouund zero to nonnegative values."""
        scale = offset = self.num_levels // 2
        return scale * codes + offset

    def nonnegative_to_codes(self, codes_nonnegative: torch.Tensor) -> torch.Tensor:
        """Convert nonnegative values to values centered arouund zero."""
        scale = offset = self.num_levels // 2
        return (codes_nonnegative - offset) / scale

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """Converts a code vector to a single index."""
        if codes.size(1) != self.dim:
            raise RuntimeError(
                f'Input code dimension {codes.size(1)} not matching the expected dimension {self.dim}, input codes shape {codes.shape}'
            )
        # convert code vectors to nonnegative values
        indices = self.codes_to_nonnegative(codes)
        # convert one nonnegative index per dimension to a single index per code vector
        indices = torch.sum(indices * self.dim_base_index, dim=1)
        return indices.to(torch.int32)

    # Implementation of VectorQuantiserBase API
    @typecheck()
    def forward(
        self, inputs: torch.Tensor, input_len: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if inputs.size(1) != self.dim:
            raise RuntimeError(
                f'Input dimension {inputs.size(1)} not matching the expected dimension {self.dim}, inputs shape {inputs.shape}'
            )

        dequantized = self.inputs_to_codes(inputs=inputs, input_len=input_len)
        indices = self.codes_to_indices(codes=dequantized)

        if input_len is not None:
            # apply masking
            dequantized = mask_sequence_tensor(dequantized, input_len)
            indices = mask_sequence_tensor(indices, input_len)

        # only 1 codebook, but return in [D, B, T] format to match RVQ API
        indices = indices.unsqueeze(0)
        return dequantized, indices

    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType(), optional=True),
        },
        output_types={"indices": NeuralType(('D', 'B', 'T'), Index())},
    )
    def encode(self, inputs: torch.Tensor, input_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convert a continuous code vector to a single index."""
        _, indices = self(inputs=inputs, input_len=input_len)
        return indices

    @typecheck(
        input_types={
            "indices": NeuralType(('D', 'B', 'T'), Index()),
            "input_len": NeuralType(tuple('B'), LengthsType(), optional=True),
        },
        output_types={
            "dequantized": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
        },
    )
    def decode(self, indices: torch.Tensor, input_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convert a single index to a continuous code vector."""
        if indices.size(0) > 1:
            # codebook dimension used for compatibility with RVQ
            raise ValueError(
                f'Expected a single codebook, got {indices.size(0)} codebooks for indices with shape {indices.shape}.'
            )

        indices = rearrange(indices, 'D B T -> B D T')
        # convert a single index to nonnegative index per-dimension
        codes_nonnegative = (indices // self.dim_base_index) % self.num_levels
        # convert nonnegative codes to codes (centered around zero)
        dequantized = self.nonnegative_to_codes(codes_nonnegative)

        if input_len is not None:
            # apply masking
            dequantized = mask_sequence_tensor(dequantized, input_len)
        return dequantized


class GroupFiniteScalarQuantizer(VectorQuantizerBase):
    """Split the input vector into groups and apply FSQ on each group separately.
    This class is for convenience. Since FSQ is applied on each group separately,
    groups can be defined arbitrarily by splitting the input vector. However, this
    class makes it easy to construct several groups with the same quantization num_levels.

    Args:
        num_groups: number of groups to split the input into, each group will be quantized separately using num_codebooks//num_groups codebooks
        codebook_dim: embedding dimension, will be split into num_groups
        **kwargs: parameters of FiniteScalarQuantizer

    References:
        Yang et al, HiFi-Codec: Group-residual Vector quantization for High Fidelity Audio Codec, 2023 (http://arxiv.org/abs/2305.02765).
    """

    def __init__(self, num_groups: int, num_levels_per_group: List[int], **kwargs):
        super().__init__()

        self.num_groups = num_groups
        self.codebook_dim_per_group = len(num_levels_per_group)

        # Initialize FSQ for each group
        self.fsqs = torch.nn.ModuleList(
            [FiniteScalarQuantizer(num_levels=num_levels_per_group, **kwargs) for _ in range(self.num_groups)]
        )

    @property
    def codebook_dim(self):
        """Input vector dimension."""
        return self.codebook_dim_per_group * self.num_groups

    @property
    def codebook_size_per_group(self):
        """Returns the size of the implicit codebook for each group."""
        return self.fsqs[0].codebook_size

    @property
    def codebook_size(self):
        """Returns the size of the implicit codebook."""
        return self.codebook_size_per_group**self.num_groups

    @typecheck()
    def forward(self, inputs, input_len):
        """Quantize each group separately, then concatenate the results."""
        inputs_grouped = inputs.chunk(self.num_groups, dim=1)

        dequantized, indices = [], []

        for in_group, fsq_group in zip(inputs_grouped, self.fsqs):
            dequantized_group, indices_group = fsq_group(inputs=in_group, input_len=input_len)
            dequantized.append(dequantized_group)
            indices.append(indices_group)

        # concatenate along the feature dimension
        dequantized = torch.cat(dequantized, dim=1)

        # concatente along the codebook dimension
        indices = torch.cat(indices, dim=0)

        return dequantized, indices

    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"indices": NeuralType(('D', 'B', 'T'), Index())},
    )
    def encode(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        """Input is split into groups, each group is encoded separately, then the results are concatenated."""
        inputs_grouped = inputs.chunk(self.num_groups, dim=1)
        indices = []

        for in_group, fsq_group in zip(inputs_grouped, self.fsqs):
            indices_group = fsq_group.encode(inputs=in_group, input_len=input_len)
            indices.append(indices_group)

        # concatenate along the codebook dimension
        indices = torch.cat(indices, dim=0)

        return indices

    @typecheck(
        input_types={
            "indices": NeuralType(('D', 'B', 'T'), Index()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "dequantized": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
        },
    )
    def decode(self, indices: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        """Input indices are split into groups, each group is decoded separately, then the results are concatenated."""
        indices_grouped = indices.chunk(self.num_groups, dim=0)
        dequantized = []

        for indices_group, fsq_group in zip(indices_grouped, self.fsqs):
            dequantized_group = fsq_group.decode(indices=indices_group, input_len=input_len)
            dequantized.append(dequantized_group)

        # concatenate along the feature dimension
        dequantized = torch.cat(dequantized, dim=1)

        return dequantized


class ResidualBlock(NeuralModule):
    """
    The residual block structure defined by the HiFi-GAN V1 and V2 configurations.

    Args:
        channels: Input dimension.
        filters: Number of channels in the residual convolutions.
        kernel_size: Kernel size of the residual convolutions.
        dilation: Dilation of the residual convolutions.
        dropout_rate: Dropout to apply to residuals.
        activation: Activation to apply in between residual convolutions.
    """

    def __init__(
        self,
        channels: int,
        filters: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout_rate: float = 0.0,
        activation: str = "lrelu",
    ):
        super(ResidualBlock, self).__init__()

        self.input_activation = CodecActivation(activation=activation, channels=channels)
        self.skip_activation = CodecActivation(activation=activation, channels=filters)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.input_conv = Conv1dNorm(
            in_channels=channels, out_channels=filters, kernel_size=kernel_size, dilation=dilation
        )
        self.skip_conv = Conv1dNorm(in_channels=filters, out_channels=channels, kernel_size=kernel_size)

    def remove_weight_norm(self):
        self.input_conv.remove_weight_norm()
        self.skip_conv.remove_weight_norm()

    @property
    def input_types(self):
        return {"inputs": NeuralType(('B', 'C', 'T'), VoidType()), "input_len": NeuralType(tuple('B'), LengthsType())}

    @property
    def output_types(self):
        return {"out": NeuralType(('B', 'C', 'T'), EncodedRepresentation())}

    @typecheck()
    def forward(self, inputs, input_len):
        conv_input = self.input_activation(inputs)
        skip_input = self.input_conv(inputs=conv_input, input_len=input_len)
        skip_input = self.skip_activation(skip_input)
        res = self.skip_conv(inputs=skip_input, input_len=input_len)
        res = self.dropout(res)
        out = inputs + res
        return out


class HiFiGANResBlock(NeuralModule):
    """
    Residual block wrapper for HiFi-GAN which creates a block for multiple dilations.

    Args:
        channels: Input dimension.
        kernel_size: Kernel size of the residual blocks.
        dilations: List of dilations. One residual block will be created for each dilation in the list.
        activation: Activation for the residual blocks.
    """

    def __init__(self, channels: int, kernel_size: int, dilations: Iterable[int], activation: str):
        super().__init__()

        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    channels=channels,
                    filters=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    activation=activation,
                )
                for dilation in dilations
            ]
        )

    def remove_weight_norm(self):
        for res_block in self.res_blocks:
            res_block.remove_weight_norm()

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'C', 'T'), VoidType()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {"out": NeuralType(('B', 'C', 'T'), VoidType())}

    @typecheck()
    def forward(self, inputs, input_len):
        out = inputs
        for res_block in self.res_blocks:
            out = res_block(inputs=out, input_len=input_len)
        return out


class HiFiGANResLayer(NeuralModule):
    """
    Residual block wrapper for HiFi-GAN which creates a block for multiple kernel sizes and dilations.
    One residual block is created for each combination of kernel size and dilation.

    Args:
        channels: Input dimension.
        kernel_sizes: List of kernel sizes.
        dilations: List of dilations.
        activation: Activation for the residual layers.

    """

    def __init__(self, channels: int, kernel_sizes: Iterable[int], dilations: Iterable[int], activation: str):
        super().__init__()

        self.res_blocks = nn.ModuleList(
            [
                HiFiGANResBlock(channels=channels, kernel_size=kernel_size, dilations=dilations, activation=activation)
                for kernel_size in kernel_sizes
            ]
        )

    def remove_weight_norm(self):
        for res_block in self.res_blocks:
            res_block.remove_weight_norm()

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'D', 'T'), VoidType()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {"out": NeuralType(('B', 'D', 'T'), VoidType())}

    @typecheck()
    def forward(self, inputs, input_len):
        residuals = [res_block(inputs=inputs, input_len=input_len) for res_block in self.res_blocks]
        out = sum(residuals) / len(residuals)
        return out


class HiFiGANEncoder(NeuralModule):
    """
    Audio encoder created by inverting the HiFi-GAN decoder.

    Args:
        encoded_dim: Dimension of encoder output.
        down_sample_rates: Rate to upsample for each decoder block. The product of the downsample rates will
            determine the output token rate. For example 2 * 2 * 8 * 8 = 256 samples per token.
        base_channels: Number of filters in the first convolution. The number of channels will be doubled after each
            downsample layer.
        in_kernel_size: Kernel size of the input convolution.
        out_kernel_size: Kernel size of the output convolution.
        resblock_kernel_sizes: List of kernel sizes to use in each residual block.
        resblock_dilation_sizes: List of dilations to use in each residual block.
        activation: Activation to use in residual and downsample layers, defaults to leaky relu.
    """

    def __init__(
        self,
        encoded_dim: int,
        down_sample_rates: Iterable[int] = (2, 2, 8, 8),
        base_channels: int = 32,
        in_kernel_size: int = 7,
        out_kernel_size: int = 7,
        resblock_kernel_sizes: Iterable[int] = (3, 7, 11),
        resblock_dilation_sizes: Iterable[int] = (1, 3, 5),
        activation: str = "lrelu",
    ):
        assert in_kernel_size > 0
        assert out_kernel_size > 0

        super().__init__()

        self.down_sample_rates = down_sample_rates
        self.pre_conv = Conv1dNorm(in_channels=1, out_channels=base_channels, kernel_size=in_kernel_size)

        in_channels = base_channels
        self.activations = nn.ModuleList([])
        self.down_sample_conv_layers = nn.ModuleList([])
        self.res_layers = nn.ModuleList([])
        for i, down_sample_rate in enumerate(self.down_sample_rates):
            res_layer = HiFiGANResLayer(
                channels=in_channels,
                kernel_sizes=resblock_kernel_sizes,
                dilations=resblock_dilation_sizes,
                activation=activation,
            )
            self.res_layers.append(res_layer)

            act = CodecActivation(activation, channels=in_channels)
            self.activations.append(act)

            out_channels = 2 * in_channels
            kernel_size = 2 * down_sample_rate

            padding = get_down_sample_padding(kernel_size=kernel_size, stride=down_sample_rate)
            down_sample_conv = Conv1dNorm(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=down_sample_rate,
                padding=padding,
            )
            in_channels = out_channels
            self.down_sample_conv_layers.append(down_sample_conv)

        self.post_activation = CodecActivation(activation, channels=in_channels)
        self.post_conv = Conv1dNorm(in_channels=in_channels, out_channels=encoded_dim, kernel_size=out_kernel_size)

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "encoded": NeuralType(('B', 'D', 'T_encoded'), EncodedRepresentation()),
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
        }

    def remove_weight_norm(self):
        self.pre_conv.remove_weight_norm()
        self.post_conv.remove_weight_norm()
        for res_layer in self.res_layers:
            res_layer.remove_weight_norm()
        for down_sample_conv in self.down_sample_conv_layers:
            down_sample_conv.remove_weight_norm()

    @typecheck()
    def forward(self, audio, audio_len):
        encoded_len = audio_len
        audio = rearrange(audio, "B T -> B 1 T")
        # [B, C, T_audio]
        out = self.pre_conv(inputs=audio, input_len=encoded_len)
        for act, res_layer, down_sample_conv, down_sample_rate in zip(
            self.activations, self.res_layers, self.down_sample_conv_layers, self.down_sample_rates
        ):
            # [B, C, T]
            out = res_layer(inputs=out, input_len=encoded_len)
            out = act(out)

            encoded_len = encoded_len // down_sample_rate
            # [B, 2 * C, T / down_sample_rate]
            out = down_sample_conv(inputs=out, input_len=encoded_len)

        out = self.post_activation(out)
        # [B, encoded_dim, T_encoded]
        encoded = self.post_conv(inputs=out, input_len=encoded_len)
        return encoded, encoded_len


class HiFiGANDecoder(NeuralModule):
    """
    Codec decoder using the HiFi-GAN generator architecture.

    Default parameters match the HiFi-GAN V1 configuration for 22.05khz.

    Args:
        input_dim: Input dimension.
        up_sample_rates: Rate to upsample for each decoder block. The product of the upsample rates should be the same
            as the overall downsample rate for your encoder. For example, a symmetric encoder/decoder can be created
            with encoder downsample rates [2, 2, 8, 8] and decoder upsample rates [8, 8, 2, 2].
        base_channels: Number of filters in the first convolution. The number of channels will be cut in
            half after each upsample layer.
        in_kernel_size: Kernel size of the input convolution.
        out_kernel_size: Kernel size of the output convolution.
        resblock_kernel_sizes: List of kernel sizes to use in each residual block.
        resblock_dilation_sizes: List of dilations to use in each residual block.
        activation: Activation to use in residual and upsample layers, defaults to leaky relu.
        output_activation: Activation to apply to output. To produce a valid audio signal, it should output values in
         the range [-1.0, 1.0]. Supports "tanh" and "clamp".
    """

    def __init__(
        self,
        input_dim: int,
        up_sample_rates: Iterable[int] = (8, 8, 2, 2),
        base_channels: int = 512,
        in_kernel_size: int = 7,
        out_kernel_size: int = 3,
        resblock_kernel_sizes: Iterable[int] = (3, 7, 11),
        resblock_dilation_sizes: Iterable[int] = (1, 3, 5),
        activation: str = "lrelu",
        output_activation: str = "tanh",
    ):
        assert in_kernel_size > 0
        assert out_kernel_size > 0

        super().__init__()

        self.up_sample_rates = up_sample_rates
        self.pre_conv = Conv1dNorm(in_channels=input_dim, out_channels=base_channels, kernel_size=in_kernel_size)

        in_channels = base_channels
        self.activations = nn.ModuleList([])
        self.up_sample_conv_layers = nn.ModuleList([])
        self.res_layers = nn.ModuleList([])
        for i, up_sample_rate in enumerate(self.up_sample_rates):
            out_channels = in_channels // 2
            kernel_size = 2 * up_sample_rate

            act = CodecActivation(activation, channels=in_channels)
            self.activations.append(act)

            up_sample_conv = ConvTranspose1dNorm(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=up_sample_rate
            )
            in_channels = out_channels
            self.up_sample_conv_layers.append(up_sample_conv)

            res_layer = HiFiGANResLayer(
                channels=in_channels,
                kernel_sizes=resblock_kernel_sizes,
                dilations=resblock_dilation_sizes,
                activation=activation,
            )
            self.res_layers.append(res_layer)

        self.post_activation = CodecActivation(activation, channels=in_channels)
        self.post_conv = Conv1dNorm(in_channels=in_channels, out_channels=1, kernel_size=out_kernel_size)
        if output_activation == "tanh":
            self.out_activation = nn.Tanh()
        elif output_activation == "clamp":
            self.out_activation = ClampActivation()
        else:
            raise ValueError(f"Invalid audio output activation {output_activation}")

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'D', 'T_encoded'), VoidType()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    def remove_weight_norm(self):
        self.pre_conv.remove_weight_norm()
        for up_sample_conv in self.up_sample_conv_layers:
            up_sample_conv.remove_weight_norm()
        for res_layer in self.res_layers:
            res_layer.remove_weight_norm()

    @typecheck()
    def forward(self, inputs, input_len):
        audio_len = input_len
        # [B, C, T_encoded]
        out = self.pre_conv(inputs=inputs, input_len=audio_len)
        for act, res_layer, up_sample_conv, up_sample_rate in zip(
            self.activations, self.res_layers, self.up_sample_conv_layers, self.up_sample_rates
        ):
            audio_len = audio_len * up_sample_rate
            out = act(out)
            # [B, C / 2, T * up_sample_rate]
            out = up_sample_conv(inputs=out, input_len=audio_len)
            out = res_layer(inputs=out, input_len=audio_len)

        out = self.post_activation(out)
        # [B, 1, T_audio]
        out = self.post_conv(inputs=out, input_len=audio_len)
        audio = self.out_activation(out)
        audio = rearrange(audio, "B 1 T -> B T")
        return audio, audio_len


class MelSpectrogramProcessor(NeuralModule):
    """
    Wrapper interface for computing mel spectrogram for codec training.
    """

    def __init__(self, sample_rate: int, win_length: int, hop_length: int, mel_dim: int = 80, log_guard: float = 1.0):
        super(MelSpectrogramProcessor, self).__init__()
        self.mel_dim = mel_dim
        self.hop_length = hop_length
        self.preprocessor = AudioToMelSpectrogramPreprocessor(
            sample_rate=sample_rate,
            highfreq=None,
            features=mel_dim,
            pad_to=1,
            exact_pad=True,
            n_window_size=win_length,
            n_window_stride=hop_length,
            window_size=False,
            window_stride=False,
            n_fft=win_length,
            mag_power=1.0,
            log=True,
            log_zero_guard_type="add",
            log_zero_guard_value=log_guard,
            mel_norm=None,
            normalize=None,
            preemph=None,
            dither=0.0,
        )

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "spec": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType()),
            "spec_len": NeuralType(tuple('B'), LengthsType()),
        }

    @typecheck()
    def forward(self, audio, audio_len):
        spec, spec_len = self.preprocessor(input_signal=audio, length=audio_len)
        return spec, spec_len


class ResNetEncoder(NeuralModule):
    """
    Residual network which uses HiFi-GAN residual blocks to encode spectrogram features without changing
    the time dimension.

    Args:
        in_channels: input dimension
        out_channels: output dimension
        num_layers: number of residual blocks to use
        hidden_channels: encoder hidden dimension
        filters: number of filters in residual block layers
        kernel_size: kernel size in residual block convolutions
        dropout_rate: Optional dropout rate to apply to residuals.
        activation: Activation to use, defaults to leaky relu.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 6,
        hidden_channels: int = 256,
        filters: int = 768,
        kernel_size: int = 3,
        dropout_rate: float = 0.1,
        activation: str = "lrelu",
    ):
        super(ResNetEncoder, self).__init__()

        self.pre_conv = Conv1dNorm(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size)
        self.res_layers = nn.ModuleList(
            [
                ResidualBlock(
                    channels=hidden_channels,
                    filters=filters,
                    kernel_size=kernel_size,
                    dropout_rate=dropout_rate,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.post_activation = CodecActivation(activation, channels=hidden_channels)
        self.post_conv = Conv1dNorm(in_channels=hidden_channels, out_channels=out_channels, kernel_size=kernel_size)

    def remove_weight_norm(self):
        self.pre_conv.remove_weight_norm()
        self.post_conv.remove_weight_norm()
        for res_layer in self.res_layers:
            res_layer.remove_weight_norm()

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'D', 'T'), VoidType()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {"encoded": NeuralType(('B', 'C', 'T'), EncodedRepresentation())}

    @typecheck()
    def forward(self, inputs, input_len):
        encoded = self.pre_conv(inputs=inputs, input_len=input_len)
        for res_layer in self.res_layers:
            encoded = res_layer(inputs=encoded, input_len=input_len)
        encoded = self.post_activation(encoded)
        encoded = self.post_conv(inputs=encoded, input_len=input_len)
        return encoded


class FullBandMelEncoder(NeuralModule):
    """
    Encoder which encodes the entire mel spectrogram with a single encoder network.

    Args:
        mel_processor: MelSpectrogramProcessor or equivalent class instance for computing the mel spectrogram from
            input audio.
        encoder: ResNetEncoder or equivalent class for encoding the mel spectrogram.
    """

    def __init__(self, mel_processor: NeuralModule, encoder: NeuralModule):
        super(FullBandMelEncoder, self).__init__()
        self.mel_processor = mel_processor
        self.encoder = encoder

    def remove_weight_norm(self):
        self.encoder.remove_weight_norm()

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "encoded": NeuralType(('B', 'C', 'T_encoded'), EncodedRepresentation()),
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
        }

    @typecheck()
    def forward(self, audio, audio_len):
        out, spec_len = self.mel_processor(audio=audio, audio_len=audio_len)
        encoded = self.encoder(inputs=out, input_len=spec_len)
        return encoded, spec_len


class MultiBandMelEncoder(NeuralModule):
    """
    Encoder which splits mel spectrogram into bands and encodes each using separate residual networks.

    Args:
        mel_bands: List of mel spectrogram bands to encode.
            Each list element is tuple of 2 elements with the start and end index of the mel features to use.
        mel_processor: MelSpectrogramProcessor or equivalent class instance for computing the mel spectrogram from
            input audio.
        encoder_kwargs: Arguments for constructing encoder for each mel band.
    """

    def __init__(self, mel_bands: Iterable[Tuple[int, int]], mel_processor: NeuralModule, **encoder_kwargs):
        super(MultiBandMelEncoder, self).__init__()
        self.validate_mel_bands(mel_dim=mel_processor.mel_dim, mel_bands=mel_bands)
        self.mel_bands = mel_bands
        self.mel_processor = mel_processor
        band_dims = [band[1] - band[0] for band in self.mel_bands]
        self.encoders = nn.ModuleList(
            [ResNetEncoder(in_channels=band_dim, **encoder_kwargs) for band_dim in band_dims]
        )

    @staticmethod
    def validate_mel_bands(mel_dim: int, mel_bands: Iterable[Tuple[int, int]]):
        mel_dims_used = np.zeros([mel_dim], dtype=bool)
        for band in mel_bands:
            mel_dims_used[band[0] : band[1]] = True

        if not all(mel_dims_used):
            missing_dims = np.where(~mel_dims_used)
            raise ValueError(f"Mel bands must cover all {mel_dim} dimensions. Missing {missing_dims}.")

        return

    def remove_weight_norm(self):
        for encoder in self.encoders:
            encoder.remove_weight_norm()

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "encoded": NeuralType(('B', 'C', 'T_encoded'), EncodedRepresentation()),
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
        }

    @typecheck()
    def forward(self, audio, audio_len):
        spec, spec_len = self.mel_processor(audio=audio, audio_len=audio_len)
        outputs = []
        for (band_start, band_end), encoder in zip(self.mel_bands, self.encoders):
            # [B, D_band, T]
            spec_band = spec[:, band_start:band_end, :]
            band_out = encoder(inputs=spec_band, input_len=spec_len)
            outputs.append(band_out)
        # [B, C, T]
        encoded = torch.cat(outputs, dim=1)
        return encoded, spec_len

class FilterbankFeatures(nn.Module):
    """Featurizer that converts wavs to Mel Spectrograms.
    See AudioToMelSpectrogramPreprocessor for args.
    """

    def __init__(
        self,
        sample_rate=16000,
        n_window_size=320,
        n_window_stride=160,
        window="hann",
        normalize="per_feature",
        n_fft=None,
        preemph=0.97,
        nfilt=64,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2**-24,
        dither=CONSTANT,
        pad_to=16,
        max_duration=16.7,
        frame_splicing=1,
        exact_pad=False,
        pad_value=0,
        mag_power=2.0,
        use_grads=False,
        rng=None,
        nb_augmentation_prob=0.0,
        nb_max_freq=4000,
        mel_norm="slaney",
        stft_exact_pad=False,  # Deprecated arguments; kept for config compatibility
        stft_conv=False,  # Deprecated arguments; kept for config compatibility
    ):
        super().__init__()
        if exact_pad and n_window_stride % 2 == 1:
            raise NotImplementedError(
                f"{self} received exact_pad == True, but hop_size was odd. If audio_length % hop_size == 0. Then the "
                "returned spectrogram would not be of length audio_length // hop_size. Please use an even hop_size."
            )
        self.log_zero_guard_value = log_zero_guard_value
        if (
            n_window_size is None
            or n_window_stride is None
            or not isinstance(n_window_size, int)
            or not isinstance(n_window_stride, int)
            or n_window_size <= 0
            or n_window_stride <= 0
        ):
            raise ValueError(
                f"{self} got an invalid value for either n_window_size or "
                f"n_window_stride. Both must be positive ints."
            )

        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.stft_pad_amount = (self.n_fft - self.hop_length) // 2 if exact_pad else None
        self.exact_pad = exact_pad

        torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'none': None,
        }
        window_fn = torch_windows.get(window, None)
        window_tensor = window_fn(self.win_length, periodic=False) if window_fn else None
        self.register_buffer("window", window_tensor)

        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = nfilt
        self.preemph = preemph
        self.pad_to = pad_to
        highfreq = highfreq or sample_rate / 2

        filterbanks = torch.tensor(
            librosa.filters.mel(
                sr=sample_rate, n_fft=self.n_fft, n_mels=nfilt, fmin=lowfreq, fmax=highfreq, norm=mel_norm
            ),
            dtype=torch.float,
        ).unsqueeze(0)
        self.register_buffer("fb", filterbanks)

        # Calculate maximum sequence length
        max_length = self.get_seq_len(torch.tensor(max_duration * sample_rate, dtype=torch.float))
        max_pad = pad_to - (max_length % pad_to) if pad_to > 0 else 0
        self.max_length = max_length + max_pad
        self.pad_value = pad_value
        self.mag_power = mag_power

        # We want to avoid taking the log of zero
        # There are two options: either adding or clamping to a small value
        if log_zero_guard_type not in ["add", "clamp"]:
            raise ValueError(
                f"{self} received {log_zero_guard_type} for the "
                f"log_zero_guard_type parameter. It must be either 'add' or "
                f"'clamp'."
            )

        self.use_grads = use_grads
        if not use_grads:
            self.forward = torch.no_grad()(self.forward)
        self._rng = random.Random() if rng is None else rng
        self.nb_augmentation_prob = nb_augmentation_prob
        if self.nb_augmentation_prob > 0.0:
            if nb_max_freq >= sample_rate / 2:
                self.nb_augmentation_prob = 0.0
            else:
                self._nb_max_fft_bin = int((nb_max_freq / sample_rate) * n_fft)

        # log_zero_guard_value is the the small we want to use, we support
        # an actual number, or "tiny", or "eps"
        self.log_zero_guard_type = log_zero_guard_type

    def stft(self, x):
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False if self.exact_pad else True,
            window=self.window.to(dtype=torch.float),
            return_complex=True,
        )

    def log_zero_guard_value_fn(self, x):
        if isinstance(self.log_zero_guard_value, str):
            if self.log_zero_guard_value == "tiny":
                return torch.finfo(x.dtype).tiny
            elif self.log_zero_guard_value == "eps":
                return torch.finfo(x.dtype).eps
            else:
                raise ValueError(
                    f"{self} received {self.log_zero_guard_value} for the "
                    f"log_zero_guard_type parameter. It must be either a "
                    f"number, 'tiny', or 'eps'"
                )
        else:
            return self.log_zero_guard_value

    def get_seq_len(self, seq_len):
        # Assuming that center is True is stft_pad_amount = 0
        pad_amount = self.stft_pad_amount * 2 if self.stft_pad_amount is not None else self.n_fft // 2 * 2
        seq_len = torch.floor_divide((seq_len + pad_amount - self.n_fft), self.hop_length) + 1
        return seq_len.to(dtype=torch.long)

    @property
    def filter_banks(self):
        return self.fb

    def forward(self, x, seq_len, linear_spec=False):
        seq_len = self.get_seq_len(seq_len)

        if self.stft_pad_amount is not None:
            x = torch.nn.functional.pad(
                x.unsqueeze(1), (self.stft_pad_amount, self.stft_pad_amount), "reflect"
            ).squeeze(1)

        # dither (only in training mode for eval determinism)
        if self.training and self.dither > 0:
            x += self.dither * torch.randn_like(x)

        # do preemphasis
        if self.preemph is not None:
            x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1)

        # disable autocast to get full range of stft values
        with torch.amp.autocast(x.device.type, enabled=False):
            x = self.stft(x)

        # torch stft returns complex tensor (of shape [B,N,T]); so convert to magnitude
        # guard is needed for sqrt if grads are passed through
        guard = 0 if not self.use_grads else CONSTANT
        x = torch.view_as_real(x)
        x = torch.sqrt(x.pow(2).sum(-1) + guard)

        if self.training and self.nb_augmentation_prob > 0.0:
            for idx in range(x.shape[0]):
                if self._rng.random() < self.nb_augmentation_prob:
                    x[idx, self._nb_max_fft_bin :, :] = 0.0

        # get power spectrum
        if self.mag_power != 1.0:
            x = x.pow(self.mag_power)

        # return plain spectrogram if required
        if linear_spec:
            return x, seq_len

        # dot with filterbank energies
        x = torch.matmul(self.fb.to(x.dtype), x)
        # log features if required
        if self.log:
            if self.log_zero_guard_type == "add":
                x = torch.log(x + self.log_zero_guard_value_fn(x))
            elif self.log_zero_guard_type == "clamp":
                x = torch.log(torch.clamp(x, min=self.log_zero_guard_value_fn(x)))
            else:
                raise ValueError("log_zero_guard_type was not understood")

        # frame splicing if required
        if self.frame_splicing > 1:
            x = splice_frames(x, self.frame_splicing)

        # normalize if required
        if self.normalize:
            x, _, _ = normalize_batch(x, seq_len, normalize_type=self.normalize)

        # mask to zero any values beyond seq_len in batch, pad to multiple of `pad_to` (for efficiency)
        max_len = x.size(-1)
        mask = torch.arange(max_len, device=x.device)
        mask = mask.repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).type(torch.bool).to(device=x.device), self.pad_value)
        del mask
        pad_to = self.pad_to
        if pad_to == "max":
            x = nn.functional.pad(x, (0, self.max_length - x.size(-1)), value=self.pad_value)
        elif pad_to > 0:
            pad_amt = x.size(-1) % pad_to
            if pad_amt != 0:
                x = nn.functional.pad(x, (0, pad_to - pad_amt), value=self.pad_value)
        return x, seq_len

class MaskedLoss(Loss):
    def __init__(self, loss_fn, loss_scale: float = 1.0):
        super(MaskedLoss, self).__init__()
        self.loss_scale = loss_scale
        self.loss_fn = loss_fn

    @property
    def input_types(self):
        return {
            "predicted": NeuralType(('B', 'D', 'T'), PredictionsType()),
            "target": NeuralType(('B', 'D', 'T'), RegressionValuesType()),
            "target_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, predicted, target, target_len):
        assert target.shape[2] == predicted.shape[2]

        # [B, D, T]
        loss = self.loss_fn(input=predicted, target=target)
        # [B, T]
        loss = torch.mean(loss, dim=1)
        # [B]
        loss = torch.sum(loss, dim=1) / torch.clamp(target_len, min=1.0)

        # [1]
        loss = torch.mean(loss)
        loss = self.loss_scale * loss

        return loss


class MaskedMAELoss(MaskedLoss):
    def __init__(self, loss_scale: float = 1.0):
        loss_fn = torch.nn.L1Loss(reduction='none')
        super(MaskedMAELoss, self).__init__(loss_fn=loss_fn, loss_scale=loss_scale)


class MaskedMSELoss(MaskedLoss):
    def __init__(self, loss_scale: float = 1.0):
        loss_fn = torch.nn.MSELoss(reduction='none')
        super(MaskedMSELoss, self).__init__(loss_fn=loss_fn, loss_scale=loss_scale)


class TimeDomainLoss(Loss):
    def __init__(self):
        super(TimeDomainLoss, self).__init__()
        self.loss_fn = MaskedMAELoss()

    @property
    def input_types(self):
        return {
            "audio_real": NeuralType(('B', 'T'), AudioSignal()),
            "audio_gen": NeuralType(('B', 'T'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, audio_real, audio_gen, audio_len):
        audio_real = rearrange(audio_real, "B T -> B 1 T")
        audio_gen = rearrange(audio_gen, "B T -> B 1 T")
        loss = self.loss_fn(target=audio_real, predicted=audio_gen, target_len=audio_len)
        return loss


class MultiResolutionMelLoss(Loss):
    """
    Multi-resolution log mel spectrogram loss.

    Args:
        sample_rate: Sample rate of audio.
        resolutions: List of resolutions, each being 3 integers ordered [num_fft, hop_length, window_length]
        mel_dims: Dimension of mel spectrogram to compute for each resolution. Should be same length as 'resolutions'.
        log_guard: Value to add to mel spectrogram to avoid taking log of 0.
    """

    def __init__(self, sample_rate: int, resolutions: List[List], mel_dims: List[int], log_guard: float = 1.0):
        super(MultiResolutionMelLoss, self).__init__()
        assert len(resolutions) == len(mel_dims)

        self.l1_loss_fn = MaskedMAELoss()
        self.l2_loss_fn = MaskedMSELoss()

        self.mel_features = torch.nn.ModuleList()
        for mel_dim, (n_fft, hop_len, win_len) in zip(mel_dims, resolutions):
            mel_feature = FilterbankFeatures(
                sample_rate=sample_rate,
                nfilt=mel_dim,
                n_window_size=win_len,
                n_window_stride=hop_len,
                n_fft=n_fft,
                pad_to=1,
                mag_power=1.0,
                log_zero_guard_type="add",
                log_zero_guard_value=log_guard,
                mel_norm=None,
                normalize=None,
                preemph=None,
                dither=0.0,
                use_grads=True,
            )
            self.mel_features.append(mel_feature)

    @property
    def input_types(self):
        return {
            "audio_real": NeuralType(('B', 'T'), AudioSignal()),
            "audio_gen": NeuralType(('B', 'T'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "l1_loss": NeuralType(elements_type=LossType()),
            "l2_loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, audio_real, audio_gen, audio_len):
        l1_loss = 0.0
        l2_loss = 0.0
        for mel_feature in self.mel_features:
            mel_real, mel_real_len = mel_feature(x=audio_real, seq_len=audio_len)
            mel_gen, _ = mel_feature(x=audio_gen, seq_len=audio_len)
            l1_loss += self.l1_loss_fn(predicted=mel_gen, target=mel_real, target_len=mel_real_len)
            l2_loss += self.l2_loss_fn(predicted=mel_gen, target=mel_real, target_len=mel_real_len)

        l1_loss /= len(self.mel_features)
        l2_loss /= len(self.mel_features)

        return l1_loss, l2_loss


class STFTLoss(Loss):
    """
    Log magnitude STFT loss.

    Args:
        resolution: Resolution of spectrogram, a list of 3 numbers ordered [num_fft, hop_length, window_length]
        log_guard: Value to add to magnitude spectrogram to avoid taking log of 0.
        sqrt_guard: Value to add to when computing absolute value of STFT to avoid NaN loss.
    """

    def __init__(self, resolution: List[int], log_guard: float = 1.0, sqrt_guard: float = 1e-5):
        super(STFTLoss, self).__init__()
        self.loss_fn = MaskedMAELoss()
        self.n_fft, self.hop_length, self.win_length = resolution
        self.register_buffer("window", torch.hann_window(self.win_length, periodic=False))
        self.log_guard = log_guard
        self.sqrt_guard = sqrt_guard

    def _compute_spectrogram(self, audio, spec_len):
        # [B, n_fft, T_spec]
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        # [B, n_fft, T_spec, 2]
        spec = torch.view_as_real(spec)
        # [B, n_fft, T_spec]
        spec_mag = torch.sqrt(spec.pow(2).sum(-1) + self.sqrt_guard)
        spec_log = torch.log(spec_mag + self.log_guard)
        spec_log = mask_sequence_tensor(spec_log, spec_len)
        return spec_log

    @property
    def input_types(self):
        return {
            "audio_real": NeuralType(('B', 'T'), AudioSignal()),
            "audio_gen": NeuralType(('B', 'T'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {"loss": NeuralType(elements_type=LossType())}

    @typecheck()
    def forward(self, audio_real, audio_gen, audio_len):
        spec_len = (audio_len // self.hop_length) + 1
        spec_real = self._compute_spectrogram(audio=audio_real, spec_len=spec_len)
        spec_gen = self._compute_spectrogram(audio=audio_gen, spec_len=spec_len)
        loss = self.loss_fn(predicted=spec_gen, target=spec_real, target_len=spec_len)
        return loss


class MultiResolutionSTFTLoss(Loss):
    """
    Multi-resolution log magnitude STFT loss.

    Args:
        resolutions: List of resolutions, each being 3 integers ordered [num_fft, hop_length, window_length]
        log_guard: Value to add to magnitude spectrogram to avoid taking log of 0.
        sqrt_guard: Value to add to when computing absolute value of STFT to avoid NaN loss.
    """

    def __init__(self, resolutions: List[List], log_guard: float = 1.0, sqrt_guard: float = 1e-5):
        super(MultiResolutionSTFTLoss, self).__init__()
        self.loss_fns = torch.nn.ModuleList(
            [STFTLoss(resolution=resolution, log_guard=log_guard, sqrt_guard=sqrt_guard) for resolution in resolutions]
        )

    @property
    def input_types(self):
        return {
            "audio_real": NeuralType(('B', 'T'), AudioSignal()),
            "audio_gen": NeuralType(('B', 'T'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {"loss": NeuralType(elements_type=LossType())}

    @typecheck()
    def forward(self, audio_real, audio_gen, audio_len):
        loss = 0.0
        for loss_fn in self.loss_fns:
            loss += loss_fn(audio_real=audio_real, audio_gen=audio_gen, audio_len=audio_len)
        loss /= len(self.loss_fns)
        return loss


class SISDRLoss(Loss):
    """
    SI-SDR loss based off of torchmetrics.functional.audio.sdr.scale_invariant_signal_distortion_ratio
    with added support for masking.
    """

    def __init__(self, epsilon: float = 1e-8):
        super(SISDRLoss, self).__init__()
        self.epsilon = epsilon

    @property
    def input_types(self):
        return {
            "audio_real": NeuralType(('B', 'T'), AudioSignal()),
            "audio_gen": NeuralType(('B', 'T'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {"loss": NeuralType(elements_type=LossType())}

    @typecheck()
    def forward(self, audio_real, audio_gen, audio_len):
        mask = get_mask_from_lengths(x=audio_real, lengths=audio_len)
        audio_len = rearrange(audio_len, 'B -> B 1')

        # Shift audio to have zero-mean
        # [B, 1]
        target_mean = torch.sum(audio_real, dim=-1, keepdim=True) / audio_len
        pred_mean = torch.sum(audio_gen, dim=-1, keepdim=True) / audio_len

        # [B, T]
        target = audio_real - target_mean
        target = target * mask
        pred = audio_gen - pred_mean
        pred = pred * mask

        # [B, 1]
        ref_pred = torch.sum(pred * target, dim=-1, keepdim=True)
        ref_target = torch.sum(target**2, dim=-1, keepdim=True)
        alpha = (ref_pred + self.epsilon) / (ref_target + self.epsilon)

        # [B, T]
        target_scaled = alpha * target
        distortion = target_scaled - pred

        # [B]
        target_scaled_power = torch.sum(target_scaled**2, dim=-1)
        distortion_power = torch.sum(distortion**2, dim=-1)

        ratio = (target_scaled_power + self.epsilon) / (distortion_power + self.epsilon)
        si_sdr = 10 * torch.log10(ratio)

        # [1]
        loss = -torch.mean(si_sdr)
        return loss


class FeatureMatchingLoss(Loss):
    """
    Standard feature matching loss measuring the difference in the internal discriminator layer outputs
    (usually leaky relu activations) between real and generated audio, scaled down by the total number of
    discriminators and layers.
    """

    def __init__(self):
        super(FeatureMatchingLoss, self).__init__()

    @property
    def input_types(self):
        return {
            "fmaps_real": [[NeuralType(elements_type=VoidType())]],
            "fmaps_gen": [[NeuralType(elements_type=VoidType())]],
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, fmaps_real, fmaps_gen):
        loss = 0.0
        for fmap_real, fmap_gen in zip(fmaps_real, fmaps_gen):
            # [B, ..., time]
            for feat_real, feat_gen in zip(fmap_real, fmap_gen):
                # [B, ...]
                diff = torch.abs(feat_real - feat_gen)
                feat_loss = torch.mean(diff) / len(fmap_real)
                loss += feat_loss

        loss /= len(fmaps_real)

        return loss


class RelativeFeatureMatchingLoss(Loss):
    """
    Relative feature matching loss as described in https://arxiv.org/pdf/2210.13438.pdf.

    This is similar to standard feature matching loss, but it scales the loss by the absolute value of
    each feature averaged across time. This might be slightly different from the paper which says the
    "mean is computed over all dimensions", which could imply taking the average across both time and
    features.

    Args:
        div_guard: Value to add when dividing by mean to avoid large/NaN values.
    """

    def __init__(self, div_guard=1e-3):
        super(RelativeFeatureMatchingLoss, self).__init__()
        self.div_guard = div_guard

    @property
    def input_types(self):
        return {
            "fmaps_real": [[NeuralType(elements_type=VoidType())]],
            "fmaps_gen": [[NeuralType(elements_type=VoidType())]],
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, fmaps_real, fmaps_gen):
        loss = 0.0
        for fmap_real, fmap_gen in zip(fmaps_real, fmaps_gen):
            # [B, ..., time]
            for feat_real, feat_gen in zip(fmap_real, fmap_gen):
                # [B, ...]
                feat_mean = torch.mean(torch.abs(feat_real), dim=-1)
                diff = torch.mean(torch.abs(feat_real - feat_gen), dim=-1)
                feat_loss = diff / (feat_mean + self.div_guard)
                # [1]
                feat_loss = torch.mean(feat_loss) / len(fmap_real)
                loss += feat_loss

        loss /= len(fmaps_real)

        return loss


class GeneratorHingedLoss(Loss):
    @property
    def input_types(self):
        return {
            "disc_scores_gen": [NeuralType(('B', 'C', 'T'), VoidType())],
        }

    @property
    def output_types(self):
        return {"loss": NeuralType(elements_type=LossType())}

    @typecheck()
    def forward(self, disc_scores_gen):
        loss = 0.0
        for disc_score_gen in disc_scores_gen:
            loss += torch.mean(F.relu(1 - disc_score_gen))

        loss /= len(disc_scores_gen)

        return loss


class GeneratorSquaredLoss(Loss):
    @property
    def input_types(self):
        return {
            "disc_scores_gen": [NeuralType(('B', 'C', 'T'), VoidType())],
        }

    @property
    def output_types(self):
        return {"loss": NeuralType(elements_type=LossType())}

    @typecheck()
    def forward(self, disc_scores_gen):
        loss = 0.0
        for disc_score_gen in disc_scores_gen:
            loss += torch.mean((1 - disc_score_gen) ** 2)

        loss /= len(disc_scores_gen)

        return loss


class DiscriminatorHingedLoss(Loss):
    @property
    def input_types(self):
        return {
            "disc_scores_real": [NeuralType(('B', 'C', 'T'), VoidType())],
            "disc_scores_gen": [NeuralType(('B', 'C', 'T'), VoidType())],
        }

    @property
    def output_types(self):
        return {"loss": NeuralType(elements_type=LossType())}

    @typecheck()
    def forward(self, disc_scores_real, disc_scores_gen):
        loss = 0.0
        for disc_score_real, disc_score_gen in zip(disc_scores_real, disc_scores_gen):
            loss_real = torch.mean(F.relu(1 - disc_score_real))
            loss_gen = torch.mean(F.relu(1 + disc_score_gen))
            loss += (loss_real + loss_gen) / 2

        loss /= len(disc_scores_real)

        return loss


class DiscriminatorSquaredLoss(Loss):
    @property
    def input_types(self):
        return {
            "disc_scores_real": [NeuralType(('B', 'C', 'T'), VoidType())],
            "disc_scores_gen": [NeuralType(('B', 'C', 'T'), VoidType())],
        }

    @property
    def output_types(self):
        return {"loss": NeuralType(elements_type=LossType())}

    @typecheck()
    def forward(self, disc_scores_real, disc_scores_gen):
        loss = 0.0
        for disc_score_real, disc_score_gen in zip(disc_scores_real, disc_scores_gen):
            loss_real = torch.mean((1 - disc_score_real) ** 2)
            loss_gen = torch.mean(disc_score_gen**2)
            loss += (loss_real + loss_gen) / 2

        loss /= len(disc_scores_real)

        return loss

@experimental
class AudioCodecModel(ModelPT):
    def __init__(self, cfg):
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)
        self.world_size = 1
        super().__init__(cfg=cfg)

        # Expected sample rate for the input audio
        self.sample_rate = cfg.sample_rate

        # Number of samples in each audio frame that is encoded
        self.samples_per_frame = cfg.samples_per_frame

        # Discriminator updates
        self.disc_updates_per_period = cfg.get("disc_updates_per_period", 1)
        self.disc_update_period = cfg.get("disc_update_period", 1)
        if self.disc_updates_per_period > self.disc_update_period:
            raise ValueError(
                f'Number of discriminator updates ({self.disc_updates_per_period}) per period must be less or equal to the configured period ({self.disc_update_period})'
            )

        # Encoder setup
        self.audio_encoder = instantiate(cfg.audio_encoder)

        # Optionally, add gaussian noise to encoder output as an information bottleneck
        encoder_noise_stdev = cfg.get("encoder_noise_stdev", 0.0)
        if encoder_noise_stdev:
            self.encoder_noise = GaussianDropout(stdev=encoder_noise_stdev)
        else:
            self.encoder_noise = None

        if "vector_quantizer" in cfg:
            self.vector_quantizer = instantiate(cfg.vector_quantizer)

            vq_output_types = list(self.vector_quantizer.output_types.keys())

            if len(vq_output_types) == 3 and vq_output_types[-1] == 'commit_loss':
                self.vector_quantizer_has_commit_loss = True
            else:
                self.vector_quantizer_has_commit_loss = False

        else:
            self.vector_quantizer = None

        # Decoder setup
        self.audio_decoder = instantiate(cfg.audio_decoder)

        # Discriminator setup
        self.discriminator = instantiate(cfg.discriminator)

        # Mel loss setup
        loss_resolutions = cfg.loss_resolutions
        mel_loss_dims = cfg.get("mel_loss_dims")
        mel_loss_log_guard = cfg.get("mel_loss_log_guard", 1.0)
        self.mel_loss_l1_scale = cfg.get("mel_loss_l1_scale", 1.0)
        self.mel_loss_l2_scale = cfg.get("mel_loss_l2_scale", 1.0)
        self.mel_loss_fn = MultiResolutionMelLoss(
            sample_rate=self.sample_rate,
            mel_dims=mel_loss_dims,
            resolutions=loss_resolutions,
            log_guard=mel_loss_log_guard,
        )

        # STFT loss setup
        stft_loss_log_guard = cfg.get("stft_loss_log_guard", 1.0)
        self.stft_loss_scale = cfg.get("stft_loss_scale", 0.0)
        self.stft_loss_fn = MultiResolutionSTFTLoss(
            resolutions=loss_resolutions,
            log_guard=stft_loss_log_guard,
        )

        # Time domain loss setup
        self.time_domain_loss_scale = cfg.get("time_domain_loss_scale", 1.0)
        self.si_sdr_loss_scale = cfg.get("si_sdr_loss_scale", 0.0)
        self.time_domain_loss_fn = TimeDomainLoss()
        self.si_sdr_loss_fn = SISDRLoss()

        # Discriminator loss setup
        self.gen_loss_scale = cfg.get("gen_loss_scale", 1.0)
        self.feature_loss_scale = cfg.get("feature_loss_scale", 1.0)
        self.gen_loss_fn = instantiate(cfg.generator_loss)
        self.disc_loss_fn = instantiate(cfg.discriminator_loss)

        feature_loss_type = cfg.get("feature_loss_type", "relative")
        if feature_loss_type == "relative":
            self.feature_loss_fn = RelativeFeatureMatchingLoss()
        elif feature_loss_type == "absolute":
            self.feature_loss_fn = FeatureMatchingLoss()
        else:
            raise ValueError(f'Unknown feature loss type {feature_loss_type}.')

        # Codebook loss setup
        if self.vector_quantizer:
            self.commit_loss_scale = cfg.get("commit_loss_scale", 1.0)
        else:
            self.commit_loss_scale = 0.0

        if self.commit_loss_scale > 0 and not self.vector_quantizer_has_commit_loss:
            raise ValueError('Commit loss is enabled but the quantizer does not support it.')

    @typecheck(
        input_types={
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "encoded": NeuralType(('B', 'D', 'T_encoded'), EncodedRepresentation()),
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
        },
    )
    def encode_audio(self, audio: torch.Tensor, audio_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply encoder on the input audio signal. Input will be padded with zeros so
        the last frame has full `self.samples_per_frame` samples.

        Args:
            audio: input time-domain signal
            audio_len: valid length for each example in the batch

        Returns:
            Encoder output `encoded` and its length in number of frames `encoded_len`
        """
        audio, audio_len = self.pad_audio(audio, audio_len)
        encoded, encoded_len = self.audio_encoder(audio=audio, audio_len=audio_len)
        return encoded, encoded_len

    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'D', 'T_encoded'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        },
    )
    def decode_audio(self, inputs: torch.Tensor, input_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply decoder on the input. Note that the input is a non-quantized encoder output or a dequantized representation.

        Args:
            inputs: encoded signal
            input_len: valid length for each example in the batch

        Returns:
            Decoded output `audio` in the time domain and its length in number of samples `audio_len`.
            Note that `audio_len` will be a multiple of `self.samples_per_frame`.
        """
        audio, audio_len = self.audio_decoder(inputs=inputs, input_len=input_len)
        return audio, audio_len

    @typecheck(
        input_types={
            "encoded": NeuralType(('B', 'D', 'T_encoded'), EncodedRepresentation()),
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"tokens": NeuralType(('B', 'C', 'T_encoded'), TokenIndex())},
    )
    def quantize(self, encoded: torch.Tensor, encoded_len: torch.Tensor) -> torch.Tensor:
        """Quantize the continuous encoded representation into a discrete
        representation for each frame.

        Args:
            encoded: encoded signal representation
            encoded_len: valid length of the encoded representation in frames

        Returns:
            A tensor of tokens for each codebook for each frame.
        """
        if not self.vector_quantizer:
            raise ValueError("Cannot quantize without quantizer")

        # vector quantizer is returning [C, B, T], where C is the number of codebooks
        tokens = self.vector_quantizer.encode(inputs=encoded, input_len=encoded_len)
        # use batch first for the output
        tokens = rearrange(tokens, 'C B T -> B C T')
        return tokens

    @typecheck(
        input_types={
            "tokens": NeuralType(('B', 'C', 'T_encoded'), TokenIndex()),
            "tokens_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "dequantized": NeuralType(('B', 'D', 'T_encoded'), EncodedRepresentation()),
        },
    )
    def dequantize(self, tokens: torch.Tensor, tokens_len: torch.Tensor) -> torch.Tensor:
        """Convert the discrete tokens into a continuous encoded representation.

        Args:
            tokens: discrete tokens for each codebook for each time frame
            tokens_len: valid length of each example in the batch

        Returns:
            Continuous encoded representation of the discrete input representation.
        """
        if not self.vector_quantizer:
            raise ValueError("Cannot dequantize without quantizer")

        # vector quantizer is using [C, B, T], where C is the number of codebooks
        tokens = rearrange(tokens, 'B C T -> C B T')
        dequantized = self.vector_quantizer.decode(indices=tokens, input_len=tokens_len)
        return dequantized

    @typecheck(
        input_types={
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "tokens": NeuralType(('B', 'C', 'T_encoded'), TokenIndex()),
            "tokens_len": NeuralType(tuple('B'), LengthsType()),
        },
    )
    def encode(self, audio: torch.Tensor, audio_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert input time-domain audio signal into a discrete representation (tokens).

        Args:
            audio: input time-domain signal, shape `(batch, number of samples)`
            audio_len: valid length for each example in the batch, shape `(batch size,)`

        Returns:
            Tokens for each codebook for each frame, shape `(batch, number of codebooks, number of frames)`,
            and the corresponding valid lengths, shape `(batch,)`
        """
        # Apply encoder to obtain a continuous vector for each frame
        encoded, encoded_len = self.encode_audio(audio=audio, audio_len=audio_len)
        # Apply quantizer to obtain discrete representation per frame
        tokens = self.quantize(encoded=encoded, encoded_len=encoded_len)
        return tokens, encoded_len

    @typecheck(
        input_types={
            "tokens": NeuralType(('B', 'C', 'T_encoded'), TokenIndex()),
            "tokens_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        },
    )
    def decode(self, tokens: torch.Tensor, tokens_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert discrete tokens into a continuous time-domain signal.

        Args:
            tokens: discrete tokens for each codebook for each time frame, shape `(batch, number of codebooks, number of frames)`
            tokens_len: valid lengths, shape `(batch,)`

        Returns:
            Decoded output `audio` in the time domain and its length in number of samples `audio_len`.
            Note that `audio_len` will be a multiple of `self.samples_per_frame`.
        """
        # Convert a discrete representation to a dequantized vector for each frame
        dequantized = self.dequantize(tokens=tokens, tokens_len=tokens_len)
        # Apply decoder to obtain time-domain audio for each frame
        audio, audio_len = self.decode_audio(inputs=dequantized, input_len=tokens_len)

        return audio, audio_len

    @typecheck(
        input_types={
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "output_audio": NeuralType(('B', 'T_audio'), EncodedRepresentation()),
            "output_audio_len": NeuralType(tuple('B'), LengthsType()),
        },
    )
    def forward(self, audio: torch.Tensor, audio_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply encoder, quantizer, decoder on the input time-domain signal.

        Args:
            audio: input time-domain signal
            audio_len: valid length for each example in the batch

        Returns:
            Reconstructed time-domain signal `output_audio` and its length in number of samples `output_audio_len`.
        """
        encoded, encoded_len = self.encode_audio(audio=audio, audio_len=audio_len)

        if self.vector_quantizer:
            # quantize to discrete tokens
            tokens = self.quantize(encoded=encoded, encoded_len=encoded_len)
            # decode tokens to audio
            output_audio, output_audio_len = self.decode(tokens=tokens, tokens_len=encoded_len)
        else:
            # no quantization, directly decode to audio
            output_audio, output_audio_len = self.decode_audio(inputs=encoded, input_len=encoded_len)

        return output_audio, output_audio_len

    def pad_audio(self, audio, audio_len):
        """Zero pad the end of the audio so that we do not have a partial end frame.
        The output will be zero-padded to have an integer number of frames of
        length `self.samples_per_frame`.

        Args:
            audio: input time-domain signal
            audio_len: valid length for each example in the batch

        Returns:
            Padded time-domain signal `padded_audio` and its length `padded_len`.
        """
        padded_len = self.samples_per_frame * torch.ceil(audio_len / self.samples_per_frame).int()
        max_len = padded_len.max().item()
        num_padding = max_len - audio.shape[1]
        padded_audio = F.pad(audio, (0, num_padding))
        return padded_audio, padded_len

    def _process_batch(self, batch):
        # [B, T_audio]
        audio = batch.get("audio")
        # [B]
        audio_len = batch.get("audio_lens")
        audio, audio_len = self.pad_audio(audio, audio_len)

        # [B, D, T_encoded]
        encoded, encoded_len = self.audio_encoder(audio=audio, audio_len=audio_len)

        if self.encoder_noise is not None:
            encoded = self.encoder_noise(encoded)

        if self.vector_quantizer:
            if self.vector_quantizer_has_commit_loss:
                encoded, _, commit_loss = self.vector_quantizer(inputs=encoded, input_len=encoded_len)
            else:
                encoded, _ = self.vector_quantizer(inputs=encoded, input_len=encoded_len)
                commit_loss = 0.0
        else:
            commit_loss = 0.0

        # [B, T]
        audio_gen, _ = self.audio_decoder(inputs=encoded, input_len=encoded_len)

        return audio, audio_len, audio_gen, commit_loss

    @property
    def disc_update_prob(self) -> float:
        """Probability of updating the discriminator."""
        return self.disc_updates_per_period / self.disc_update_period

    def should_update_disc(self, batch_idx) -> bool:
        """Decide whether to update the descriminator based
        on the batch index and configured discriminator update period.
        """
        disc_update_step = batch_idx % self.disc_update_period
        return disc_update_step < self.disc_updates_per_period

    def setup_training_data(self):
        ...
    
    def setup_validation_data(self):
        ...

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        models = []

        model = PretrainedModelInfo(
            pretrained_model_name="audio_codec_16khz_small",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/audio_codec_16khz_small/versions/v1/files/audio_codec_16khz_small.nemo",
            description="For details about this model please refer to the model card: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/audio_codec_16khz_small",
        )
        models.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="mel_codec_22khz_medium",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/mel_codec_22khz_medium/versions/v1/files/mel_codec_22khz_medium.nemo",
            description="For details about this model please refer to the model card: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/mel_codec_22khz_medium",
        )
        models.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="mel_codec_44khz_medium",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/mel_codec_44khz_medium/versions/v1/files/mel_codec_44khz_medium.nemo",
            description="For details about this model please refer to the model card: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/mel_codec_44khz_medium",
        )
        models.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="mel_codec_22khz_fullband_medium",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/mel_codec_22khz_fullband_medium/versions/v1/files/mel_codec_22khz_fullband_medium.nemo",
            description="For details about this model please refer to the model card: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/mel_codec_22khz_fullband_medium",
        )
        models.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="mel_codec_44khz_fullband_medium",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/mel_codec_44khz_fullband_medium/versions/v1/files/mel_codec_44khz_fullband_medium.nemo",
            description="For details about this model please refer to the model card: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/mel_codec_44khz_fullband_medium",
        )
        models.append(model)

        return models