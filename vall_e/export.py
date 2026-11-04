# Release checkpoints: a training checkpoint (last.pt: weights, optimizer state, config, vocabulary) exported to a
# safetensors file with just what inference needs (fp16 weights; the config and phoneme vocabulary in its metadata),
# and loading either kind. The default model downloads on first use.
#
#   python -m vall_e.export training/v3-main6-len/last.pt data/models/nemo-44khz-nar-flow-225m.sft
import argparse
import json
import os
from pathlib import Path

import torch

DEFAULT_MODEL = "nemo-44khz-nar-flow-225m.sft"
DEFAULT_URL = os.environ.get("VALL_E_MODEL_URL", "https://huggingface.co/ecker/vall-e/resolve/main/models/ckpt/nemo-44khz-nar-flow-225m/fp16.sft")
MODEL_DIR = Path(os.environ.get("VALL_E_MODELS", Path(__file__).resolve().parents[1] / "data" / "models"))
FORMAT = "vall-e-v3"


def export(ckpt, out, dtype=torch.float16):
	"""ckpt (a training checkpoint) -> out (.sft): the model weights in `dtype`, config and vocabulary as metadata."""
	from safetensors.torch import save_file
	st = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
	weights = {k: v.to(dtype).contiguous() if v.is_floating_point() else v.contiguous() for k, v in st["model"].items()}
	meta = dict(format=FORMAT, config=json.dumps(st["config"]), vocab=json.dumps(st["vocab"], ensure_ascii=False),
		source=str(ckpt), step=str(st.get("step", "")))
	Path(out).parent.mkdir(parents=True, exist_ok=True)
	save_file(weights, str(out), metadata=meta)
	return out


def load_checkpoint(path):
	"""A training checkpoint (.pt) or an export (.sft / .safetensors) -> (state dict, config, vocabulary). None (or the
	default model's name) gives the default model, downloaded on first use."""
	if path is None or str(path) == DEFAULT_MODEL:
		path = default_model()
	path = Path(path)
	if path.suffix in (".sft", ".safetensors"):
		from safetensors import safe_open
		with safe_open(str(path), framework="pt") as f:
			meta = f.metadata()
			if meta.get("format") != FORMAT:
				raise ValueError(f"{path} isn't a {FORMAT} export (format: {meta.get('format')})")
			state = {}
			for k in f.keys():
				t = f.get_tensor(k)
				state[k] = t.float() if t.is_floating_point() else t
		return state, json.loads(meta["config"]), json.loads(meta["vocab"])
	st = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
	return st["model"], st["config"], st["vocab"]


def default_model():
	"""The default model's local path, downloading it from DEFAULT_URL ($VALL_E_MODEL_URL) the first time."""
	path = MODEL_DIR / DEFAULT_MODEL
	if not path.exists():
		import requests
		from tqdm import tqdm
		path.parent.mkdir(parents=True, exist_ok=True)
		with requests.get(DEFAULT_URL, stream=True) as r:
			r.raise_for_status()
			bar = tqdm(total=int(r.headers.get("content-length", 0)), unit="B", unit_scale=True, desc=f"downloading {DEFAULT_MODEL}")
			with open(path.with_suffix(".tmp"), "wb") as f:
				for chunk in r.iter_content(1 << 20):
					f.write(chunk); bar.update(len(chunk))
		path.with_suffix(".tmp").replace(path)
	return path


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("ckpt", type=Path, help="a training checkpoint (last.pt)")
	ap.add_argument("out", type=Path, help="the export (.sft)")
	ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
	args = ap.parse_args()
	export(args.ckpt, args.out, getattr(torch, args.dtype))
	print(f"wrote {args.out} ({args.out.stat().st_size / 2**20:.0f} MB)")


if __name__ == "__main__":
	main()
