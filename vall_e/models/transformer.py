"""
# https://github.com/enhuiz/vall-e/
"""

import math
import torch
import torch.nn.functional as F
import traceback

from typing import Literal, overload
from functools import partial
from einops import rearrange

from torch import Tensor, einsum, nn
from torch.utils.checkpoint import checkpoint

from ..utils import wrapper as ml
from .adaln import AdaLN

class SinusoidalEmbedding(nn.Module):
	def __init__(self, d_model):
		super().__init__()
		self.d_model = d_model
		exponent = torch.arange(self.d_half, dtype=torch.float32)
		exponent = exponent / self.d_half
		omega = torch.exp(-math.log(1e4) * exponent)
		self.omega: torch.Tensor
		self.register_buffer("omega", omega, persistent=False)

	@property
	def d_half(self):
		assert self.d_model % 2 == 0, "Only support even d_model."
		return self.d_model // 2

	def forward(self, x):
		"""
		Args:
			x: (...)
		Returns:
			pe: (... d)
		"""
		omega = self.omega

		while omega.dim() <= x.dim():
			omega = omega.unsqueeze(0)  # (... d)

		x = x.unsqueeze(-1)  # (... 1)
		x = omega * x
		x = torch.cat([x.sin(), x.cos()], dim=-1)

		return x

	def get_pe(self, n: int):
		"""
		Args:
			n: int
		Returns:
			pe: (n d)
		"""
		device = self.omega.device
		return self.forward(torch.arange(n, device=device))

	def add_pe(self, x):
		"""
		Args:
			x: (b t c)
		"""
		e = self.get_pe(x.shape[1])  # t d
		e = e[None]  # b t d
		x = x + e
		return x


class Attention(nn.Module):
	def __init__(self, d_model, n_heads, causal):
		super().__init__()
		assert d_model % n_heads == 0
		dim_head = d_model // n_heads
		self.causal = causal
		self.n_heads = n_heads
		self.scale = dim_head**-0.5

		self.to_qkv = nn.Linear(d_model, d_model * 3, bias=False)
		self.to_out = nn.Linear(d_model, d_model)

	def forward(self, x, m):
		"""
		Args:
			x: (b t c)
			m: (b t c), 1 is data, 0 is padding
		Returns:
			x: (b t c)
		"""
		h = self.n_heads

		q, k, v = self.to_qkv(x).chunk(3, dim=-1)
		q, k, v = map(lambda t: rearrange(t, "b t (h d) -> b t h d", h=h), (q, k, v))

		e = einsum("b i h d, b j h d -> b i j h", q, k)
		e = e * self.scale

		kpm = m.unsqueeze(1) * m.unsqueeze(2)  # b i j 1

		if self.causal:
			with ml.autocast(kpm, torch.bfloat16, torch.float16) as k:
				kpm = k.squeeze(-1).tril().unsqueeze(-1)  # b i j 1

		e = e.masked_fill(kpm == 0, -torch.finfo(e.dtype).max)
		a = e.softmax(dim=2)  # Normalize on j, i.e. key

		o = einsum("b i j h, b j h d -> b i h d", a, v)
		o = o.flatten(-2)
		o = self.to_out(o)  # b t c

		o = o * m

		return o

class PrenormResidual(nn.Module):
	def __init__(
		self,
		block,
		d_model,
		p_dropout,
		requires_mask=False,
		norm_type="ln",
		n_levels: int | None = None,
	):
		super().__init__()
		self.block = block
		self.requires_mask = requires_mask
		self.norm_type = norm_type
		if norm_type == "ln":
			self.norm = nn.LayerNorm(d_model)
		elif norm_type == "adaln":
			assert n_levels is not None
			self.norm = AdaLN(d_model, n_levels)
		else:
			raise NotImplementedError(norm_type)
		self.dropout = nn.Dropout(p_dropout)

	def forward(self, x, m, l):
		"""
		Args:
			x: input (b t d)
			m: mask (b t 1), 1 is valuable and 0 is padding
			l: level to use, required only for AdaLN
		"""
		nopts = {"l": l} if self.norm_type == "adaln" else {}
		bopts = {"m": m} if self.requires_mask else {}
		x = x + self.dropout(self.block(self.norm(x, **nopts) * m, **bopts))
		return x * m


class Block(nn.Sequential):
	def __init__(self, d_model, n_heads, p_dropout, causal, norm_type, n_levels):
		super().__init__()

		self.attn = PrenormResidual(
			Attention(d_model, n_heads, causal),
			d_model=d_model,
			p_dropout=p_dropout,
			requires_mask=True,
			norm_type=norm_type,
			n_levels=n_levels,
		)

		n_ff = d_model * 4 # 1024 * 4 = 4096 feed-forwards
		self.ffn = PrenormResidual(
			nn.Sequential(
				nn.Linear(d_model, n_ff),
				nn.GELU(),
				nn.Dropout(p_dropout),
				nn.Linear(n_ff, d_model),
			),
			d_model=d_model,
			p_dropout=p_dropout,
			norm_type=norm_type,
			n_levels=n_levels,
		)

	def forward(self, x, m, l):
		"""
		Args:
			x: (b t c)
			m: (b t 1)
			l: (b)
		"""
		poor_in_vram = True
		if x.requires_grad and poor_in_vram:
			x = checkpoint(self.attn, x, m, l, use_reentrant=False)
		else:
			x = self.attn(x, m, l)
		x = self.ffn(x, m, l)
		return x