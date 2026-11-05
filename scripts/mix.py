# WiSE-FT weight mixing (how v1.2 was made: a = 0.7 of training/v3-long-len with v3-main6-avg3-len, then a new length head).
#   python scripts/mix.py <fine-tuned ckpt> <base ckpt> <a> <out>
import sys, torch
ft, base, a, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
F = torch.load(ft, map_location="cpu", weights_only=False); B = torch.load(base, map_location="cpu", weights_only=False)["model"]
# the length heads are unrelated (each trained from a fresh init on its own backbone): only the backbone is mixed
head = lambda k: k == "len_q" or k.startswith("length.")
m = {k: (a * v.float() + (1 - a) * B[k].float()) if v.is_floating_point() and not head(k) else v for k, v in F["model"].items()}
torch.save(dict(model=m, config=F["config"], vocab=F["vocab"], step=f"soup {a} of {ft} + {base}"), out)
print("wrote", out)
