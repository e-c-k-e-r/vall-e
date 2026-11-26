# PyTorch's expandable-segment allocator (less fragmentation with varying batch shapes) where it works: set before the
# first GPU allocation, which reads it. HIP 7.15 (torch's ROCm 10 wheels) hangs that first allocation with it, where 6.3
# and 7.1 run with it; a setting already in the environment wins.
import os


def expandable_segments():
	import torch
	v = torch.version.hip
	if v and tuple(int(x) for x in v.split(".")[:2]) >= (7, 15):
		return
	os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
