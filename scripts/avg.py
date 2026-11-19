# Checkpoint averaging: the mean of several checkpoints of one run (how v1.1's avg3 was made). Length heads are dropped
# (train a new one on the average, as for mix.py's soups).
#   python scripts/avg.py <out> <ckpt> <ckpt> [...]
import sys, torch
out, paths = sys.argv[1], sys.argv[2:]
head = lambda k: k == "len_q" or k.startswith("length.")
acc, first = None, None
for p in paths:
	c = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
	m = {k: v for k, v in c["model"].items() if not head(k)}
	if acc is None:
		first = c; acc = {k: v.double() if v.is_floating_point() else v for k, v in m.items()}
	else:
		for k, v in m.items():
			if v.is_floating_point():
				acc[k] += v.double()
	del c
model = {k: (v / len(paths)).to(first["model"][k].dtype) if v.is_floating_point() else v for k, v in acc.items()}
torch.save(dict(model=model, config=first["config"], vocab=first["vocab"], step=f"mean of {paths}"), out)
print("wrote", out)
