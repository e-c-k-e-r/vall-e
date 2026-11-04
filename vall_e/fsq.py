# nvidia/audio-codec-44khz quantizes a frame as 8 groups of 4 scalars at [8, 5, 5, 5] levels (1000 codes per group).
# These convert between its code indices, per-scalar level indices ("digits") and the [-1, 1] values its decoder sees,
# matching FiniteScalarQuantizer in vall_e/emb/nemo.py.
import torch

GROUPS = 8
LEVELS = [8, 5, 5, 5]
SCALARS = GROUPS * len(LEVELS)  # 32 values per frame
MAX_LEVEL = max(LEVELS)

_levels = torch.tensor(LEVELS * GROUPS)  # [32]
_base = torch.cumprod(torch.tensor([1] + LEVELS[:-1]), 0)  # [1, 8, 40, 200]


def codes_to_digits(codes):
	"""[8, T] code indices -> [T, 32] level indices."""
	codes = codes.long().t()  # [T, 8]
	digits = (codes.unsqueeze(-1) // _base.to(codes.device)) % torch.tensor(LEVELS, device=codes.device)  # [T, 8, 4]
	return digits.flatten(1)


def digits_to_codes(digits):
	"""[T, 32] level indices -> [8, T] code indices."""
	return (digits.long().view(-1, GROUPS, len(LEVELS)) * _base.to(digits.device)).sum(-1).t()


def digits_to_values(digits):
	"""[..., k * 32] level indices (k frames per patch) -> values in [-1, 1], same shape."""
	half = (_levels.to(digits.device) // 2).float().repeat(digits.shape[-1] // SCALARS)
	return (digits.float() - half) / half


def level_mask(device=None):
	"""[32, MAX_LEVEL] bool, True where a scalar has that level."""
	return torch.arange(MAX_LEVEL, device=device)[None, :] < _levels.to(device)[:, None]
