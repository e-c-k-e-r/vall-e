# "borrowed" with love from https://github.com/MadsToftrup/Apollo-dev/blob/main/galore_torch/apollo.py
# to be replaced with the official implementation (https://github.com/zhuhanqing/APOLLO) maybe

import torch
import math
import numpy as np

from torch import nn
from torch.optim import Optimizer

from typing import Any, Callable, Dict, Generator, Iterable, Optional, Sequence, Union, Tuple

from transformers.utils.versions import require_version

class GaLoreProjector:
	def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type='std'):
		self.rank = rank
		self.verbose = verbose
		self.update_proj_gap = update_proj_gap
		self.scale = scale
		self.ortho_matrix = None
		self.proj_type = proj_type
		self.svd_count = 0

	def project(self, full_rank_grad, iter):

		if self.ortho_matrix is not None and self.ortho_matrix.device != full_rank_grad.device:
			self.ortho_matrix = self.ortho_matrix.to(full_rank_grad.device)

		if self.proj_type == 'std':
			if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
				if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
					self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
					self.svd_count += 1
				low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
			else:
				if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
					self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
					self.svd_count += 1
				low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
		elif self.proj_type == 'reverse_std':
			if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
				if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
					self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
					self.svd_count += 1
				low_rank_grad = torch.matmul(self.ortho_matrix.t(),full_rank_grad)
			else:
				if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
					self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
					self.svd_count += 1
				low_rank_grad = torch.matmul(full_rank_grad,self.ortho_matrix.t())
		elif self.proj_type == 'right':
			if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
				self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
				self.svd_count += 1
			low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
		elif self.proj_type == 'left':
			if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
				self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
				self.svd_count += 1
			low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
		elif self.proj_type == 'full':
			if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
				self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='full')
				self.svd_count += 1
			low_rank_grad = torch.matmul(self.ortho_matrix[0].t(), full_rank_grad) @ self.ortho_matrix[1].t()
				
		return low_rank_grad

	def project_back(self, low_rank_grad):

		if self.proj_type == 'std':
			if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
				full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
			else:
				full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
		elif self.proj_type == 'reverse_std':
			if low_rank_grad.shape[0] <= low_rank_grad.shape[1]: # note this is different from std
				full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
			else:
				full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
		elif self.proj_type == 'right':
			full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
		elif self.proj_type == 'left':
			full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
		elif self.proj_type == 'full':
			full_rank_grad = torch.matmul(self.ortho_matrix[0], low_rank_grad) @ self.ortho_matrix[1]
		return full_rank_grad * self.scale

	# svd decomposition
	def get_orthogonal_matrix(self, weights, rank, type):
		module_params = weights

		if module_params.data.dtype != torch.float:
			float_data = False
			original_type = module_params.data.dtype
			original_device = module_params.data.device
			matrix = module_params.data.float()
		else:
			float_data = True
			matrix = module_params.data
			
		U, s, Vh = torch.linalg.svd(matrix, full_matrices = False)
		
		#make the smaller matrix always to be orthogonal matrix
		if type=='right':
			A = U[:, :rank] @ torch.diag(s[:rank])
			B = Vh[:rank, :]
			
			if not float_data:
				B = B.to(original_device).type(original_type)
			return B
		elif type=='left':
			A = U[:, :rank]
			B = torch.diag(s[:rank]) @ Vh[:rank, :]
			if not float_data:
				A = A.to(original_device).type(original_type)
			return A
		elif type=='full':
			A = U[:, :rank]
			B = Vh[:rank, :]
			if not float_data:
				A = A.to(original_device).type(original_type)
				B = B.to(original_device).type(original_type)
			return [A, B]
		else:
			raise ValueError('type should be left, right or full')

def stable_randn(
	shape: Union[int, Sequence[int]],
	seed: int,
	device: Optional[Union[str, torch.device]] = None,
	dtype: Optional[torch.dtype] = torch.float32,
):
	if device is None:
		device = torch.device("cpu")
	generator = torch.Generator(device=device).manual_seed(seed)
	rn = torch.randn(shape, generator=generator, device=generator.device, dtype=dtype)
	return rn


def next_seed(seed: int, adv: int = 0xF):
	"""
	This is a naive helper function to generate a new seed from the given seed.
	"""
	generator = torch.Generator().manual_seed(seed)
	return torch.randint(
		0, torch.iinfo(torch.int64).max, (adv,), generator=generator, device=generator.device
	).tolist()[-1]


def split_seed(seed: int):
	generator = torch.Generator().manual_seed(seed)
	return tuple(
		torch.randint(0, torch.iinfo(torch.int64).max, (2,), generator=generator, device=generator.device).tolist()
	)


class GradientProjector:
	def __init__(
		self, rank, update_proj_gap=200, alpha=1.0, proj_type="std", seed=0
	):
		# This is a lazy implementation as we store the projection matrix instead of re-generation every iteration
		self.rank = rank
		self.update_proj_gap = update_proj_gap
		self.alpha = alpha
		self.proj_type = proj_type

		self.ortho_matrix = None
		self.seed = seed

	def project(self, full_rank_grad, iter):

		if self.proj_type == "std":
			if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
				if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
					self.ortho_matrix = self.get_orthogonal_matrix(
						full_rank_grad, self.rank, type="right", seed=self.seed
					)
					self.seed = next_seed(self.seed)
				low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
			else:
				if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
					self.ortho_matrix = self.get_orthogonal_matrix(
						full_rank_grad, self.rank, type="left", seed=self.seed
					)
					self.seed = next_seed(self.seed)
					
				low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
		elif self.proj_type == "reverse_std":
			if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
				if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
					self.ortho_matrix = self.get_orthogonal_matrix(
						full_rank_grad, self.rank, type="left", seed=self.seed
					)
					self.seed = next_seed(self.seed)
					
				low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
			else:
				if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
					self.ortho_matrix = self.get_orthogonal_matrix(
						full_rank_grad, self.rank, type="right", seed=self.seed
					)
					self.seed = next_seed(self.seed)
				low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
		elif self.proj_type == "right":
			if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
				self.ortho_matrix = self.get_orthogonal_matrix(
					full_rank_grad, self.rank, type="right", seed=self.seed
				)
				self.seed = next_seed(self.seed)
			low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
		elif self.proj_type == "left":
			if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
				self.ortho_matrix = self.get_orthogonal_matrix(
					full_rank_grad, self.rank, type="left", seed=self.seed
				)
				self.seed = next_seed(self.seed)
			low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
		elif self.proj_type == "full":
			if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
				self.ortho_matrix = self.get_orthogonal_matrix(
					full_rank_grad, self.rank, type="full", seed=self.seed
				)
				self.seed = next_seed(self.seed)
			low_rank_grad = (
				torch.matmul(self.ortho_matrix[0].t(), full_rank_grad)
				@ self.ortho_matrix[1].t()
			)

		return low_rank_grad

	# random low rank projection
	def get_orthogonal_matrix(self, weights, rank, type, seed):
		module_params = weights

		if module_params.data.dtype != torch.float:
			float_data = False
			original_type = module_params.data.dtype
			original_device = module_params.data.device
			matrix = module_params.data.float()
		else:
			float_data = True
			matrix = module_params.data

		if type == "left":
			proj = stable_randn(
				(matrix.shape[0], rank), seed=seed, device=matrix.device, dtype=matrix.dtype
			) / math.sqrt(rank)
			if not float_data:
				proj = proj.to(original_device).type(original_type)
			return proj
		elif type == "right":
			proj = stable_randn(
				(rank, matrix.shape[1]), seed=seed, device=matrix.device, dtype=matrix.dtype
			) / math.sqrt(rank)
			if not float_data:
				proj = proj.to(original_device).type(original_type)
			return proj
		elif type == "full":
			raise NotImplementedError("full rank projection is not implemented yet")
		else:
			raise ValueError("type should be left, right or full")

class Apollo(Optimizer):
	"""
	Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
	Regularization](https://arxiv.org/abs/1711.05101).

	Parameters:
		params (`Iterable[nn.parameter.Parameter]`):
			Iterable of parameters to optimize or dictionaries defining parameter groups.
		lr (`float`, *optional*, defaults to 0.001):
			The learning rate to use.
		betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
			Adam's betas parameters (b1, b2).
		eps (`float`, *optional*, defaults to 1e-06):
			Adam's epsilon for numerical stability.
		weight_decay (`float`, *optional*, defaults to 0.0):
			Decoupled weight decay to apply.
		correct_bias (`bool`, *optional*, defaults to `True`):
			Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
		no_deprecation_warning (`bool`, *optional*, defaults to `False`):
			A flag used to disable the deprecation warning (set to `True` to disable the warning).
	"""

	def __init__(
		self,
		params: Iterable[nn.parameter.Parameter],
		lr: float = 1e-3,
		betas: Tuple[float, float] = (0.9, 0.999),
		eps: float = 1e-6,
		weight_decay: float = 0.0,
		correct_bias: bool = True,
		scale_front: bool = False,
	):
		if lr < 0.0:
			raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
		if not 0.0 <= betas[0] < 1.0:
			raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
		if not 0.0 <= betas[1] < 1.0:
			raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
		if not 0.0 <= eps:
			raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
		defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
		super().__init__(params, defaults)
		
		self.scale_front = scale_front

		params_idx = 0
		for group in self.param_groups:
			for p in group["params"]:
				params_idx += 1
				if p.requires_grad:
					self.state[p]["seed"] = params_idx

	@torch.no_grad()
	def step(self, closure: Callable = None):
		"""
		Performs a single optimization step.

		Arguments:
			closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
		"""
		loss = None
		if closure is not None:
			loss = closure()

		params_idx = 0
		for group in self.param_groups:
			for p in group["params"]:
				params_idx += 1
				if p.grad is None:
					continue
				grad = p.grad
				if grad.is_sparse:
					raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

				state = self.state[p]
				
				if "step" not in state:
					state["step"] = 0

				if "seed" not in state:
					state["seed"] = params_idx

				# GaLore Projection
				if "rank" in group:
					if "projector" not in state:
						if group["proj"] == "random":
							state["projector"] = GradientProjector(group["rank"], 
								update_proj_gap=group["update_proj_gap"], 
								alpha=group["scale"], 
								proj_type=group["proj_type"],
								seed=state["seed"])

						elif group["proj"] == "svd":
							state["projector"] = GaLoreProjector(group["rank"], 
								update_proj_gap=group["update_proj_gap"], 
								scale=group["scale"], 
								proj_type=group["proj_type"])

					grad = state["projector"].project(grad, state["step"])

				# State initialization
				if "exp_avg" not in state:
					# Exponential moving average of gradient values
					state["exp_avg"] = torch.zeros_like(grad)
					# Exponential moving average of squared gradient values
					state["exp_avg_sq"] = torch.zeros_like(grad)

				exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
				beta1, beta2 = group["betas"]

				state["step"] += 1

				# Decay the first and second moment running average coefficient
				# In-place operations to update the averages at the same time
				exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
				exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
				denom = exp_avg_sq.sqrt().add_(group["eps"])

				step_size = group["lr"]
				if group["correct_bias"]:  # No bias correction for Bert
					bias_correction1 = 1.0 - beta1 ** state["step"]
					bias_correction2 = 1.0 - beta2 ** state["step"]
					step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

				# compute norm gradient
				norm_grad = exp_avg / denom

				if "rank" in group:
					if group['scale_type'] == 'channel':
						norm_dim = 0 if norm_grad.shape[0] < norm_grad.shape[1] else 1
						scaling_factor = (
							torch.norm(norm_grad, dim=norm_dim) /
							(torch.norm(grad, dim=norm_dim) + 1e-8)
						)
						if norm_dim == 1:
							scaling_factor = scaling_factor.unsqueeze(1)

					elif group['scale_type'] == 'tensor':
						scaling_factor = (
							torch.norm(norm_grad) /
							(torch.norm(grad) + 1e-8)
						)

					scaling_grad = p.grad * scaling_factor

					# Use Norm-Growth Limiter in Fira
					if "scaling_grad" in state:
						scaling_grad_norm = torch.norm(scaling_grad)
						limiter = max(
								scaling_grad_norm / 
								(state["scaling_grad"] + 1e-8),
								1.01,
							) / 1.01
						scaling_grad = scaling_grad / limiter
						state["scaling_grad"] = scaling_grad_norm / limiter
					else:
						state["scaling_grad"] = torch.norm(scaling_grad)

					norm_grad = scaling_grad * np.sqrt(group["scale"])

				p.add_(norm_grad, alpha=-step_size)

				# Just adding the square of the weights to the loss function is *not*
				# the correct way of using L2 regularization/weight decay with Adam,
				# since that will interact with the m and v parameters in strange ways.
				#
				# Instead we want to decay the weights in a manner that doesn't interact
				# with the m/v parameters. This is equivalent to adding the square
				# of the weights to the loss with plain (non-momentum) SGD.
				# Add weight decay at the end (fixed version)
				if group["weight_decay"] > 0.0:
					p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

		return loss