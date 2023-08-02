"""
# https://github.com/enhuiz/vall-e/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaLN(nn.Module):
	def __init__(self, d_model, n_levels, eps=1e-5, k=0.1, c=2):
		super().__init__()
		self.eps = eps
		self.emb = nn.Embedding(n_levels, d_model * 2)
		self.k = k
		self.c = c
		nn.init.zeros_(self.emb.weight)

	def forward(self, x, l):
		h = F.layer_norm(x, x.shape[-1:], eps=self.eps)

		# The initial implementation (https://github.com/enhuiz/vall-e/blob/fbf023448c08e55c0422eefed7fc234cf8b76680/vall_e/vall_e/base.py#L135)
		# performed worse than vanilla LayerNorm.
		# The authors mentioned another AdaNorm paper (https://openreview.net/pdf?id=HyxndNrxLB) as they introduce AdaLN.
		# Did they use AdaNorm inside AdaLN? (as follows)
		h = self.c * (1 - (self.k * h).detach()) * h

		logγ, β = self.emb(l).unsqueeze(1).chunk(2, dim=-1)
		y = logγ.exp() * h + β

		return y