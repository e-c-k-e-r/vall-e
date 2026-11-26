# Release checkpoints: a training checkpoint (last.pt: weights, optimizer state, config, vocabulary) exported to a
# safetensors file with just what inference needs (fp16 weights; the config and phoneme vocabulary in its metadata),
# and loading either kind. The default model downloads on first use.
#
#   python -m vall_e.export training/v3-main6-len/last.pt data/models/nemo-44khz-nar-flow-225m.sft
#   python -m vall_e.export codec data/models/nemo-44khz-codec.sft      (the codec, for engines)
import argparse
import json
import os
from pathlib import Path

import torch

DEFAULT_MODEL = "nemo-44khz-nar-flow-225m.sft"
DEFAULT_URL = os.environ.get("VALL_E_MODEL_URL", "https://huggingface.co/ecker/vall-e/resolve/main/models/ckpt/nemo-44khz-nar-flow-225m/fp16.sft")
MODEL_DIR = Path(os.environ.get("VALL_E_MODELS", Path(__file__).resolve().parents[1] / "data" / "models"))
FORMAT = "vall-e-v3"


def export(ckpt, out, dtype=torch.float16, cfg=None, cfg_prompt=None):
	"""ckpt (a training checkpoint) -> out (.sft): the model weights in `dtype`, config and vocabulary as metadata, and
	the guidance it was tuned for (cfg, cfg_prompt; tts.py's defaults when given: v2.0 3.0 / 2.0, a distilled model 0 / 0)."""
	from safetensors.torch import save_file
	st = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
	weights = {k: v.to(dtype).contiguous() if v.is_floating_point() else v.contiguous() for k, v in st["model"].items()}
	meta = dict(format=FORMAT, config=json.dumps(st["config"]), vocab=json.dumps(st["vocab"], ensure_ascii=False),
		source=str(ckpt), step=str(st.get("step", "")))
	if cfg is not None:
		meta["guidance"] = json.dumps(dict(cfg=cfg, cfg_prompt=cfg_prompt if cfg_prompt is not None else 2.0))
	Path(out).parent.mkdir(parents=True, exist_ok=True)
	save_file(weights, str(out), metadata=meta)
	return out


def export_codec(out):
	"""The NeMo 44 kHz codec (HiFi-GAN encoder and decoder, FSQ between) for engines: FP32 weights with the weight norm
	folded in (NeMo's own remove_weight_norm) under audio_encoder. / audio_decoder., the layout of both (channels, sample
	rates, residual kernels and dilations, activations, FSQ levels) in the metadata."""
	from safetensors.torch import save_file
	from .evalset import load_codec
	codec = load_codec("cpu")
	enc, dec, q = codec.audio_encoder, codec.audio_decoder, codec.vector_quantizer
	enc.remove_weight_norm()
	dec.remove_weight_norm()
	dec.post_conv.remove_weight_norm()  # HiFiGANDecoder.remove_weight_norm leaves the output conv's
	weights = {f"{n}.{k}": v.float().contiguous() for n, m in (("audio_encoder", enc), ("audio_decoder", dec)) for k, v in m.state_dict().items()}
	assert not any(k.endswith(("weight_g", "weight_v")) for k in weights)
	res = lambda layer: dict(resblock_kernel_sizes=[b.res_blocks[0].input_conv.conv.kernel_size[0] for b in layer.res_blocks],
		resblock_dilation_sizes=[r.input_conv.conv.dilation[0] for r in layer.res_blocks[0].res_blocks])
	act = lambda a: {"HalfSnake": "half_snake", "LeakyReLU": "lrelu"}[type(a.activation).__name__]
	config = dict(sample_rate=int(codec.sample_rate), samples_per_frame=int(codec.samples_per_frame),
		levels=[int(x) for x in q.fsqs[0].num_levels.flatten()], groups=int(q.num_groups), fsq_eps=float(q.fsqs[0].eps),
		encoder=dict(base_channels=enc.pre_conv.conv.out_channels, encoded_dim=enc.post_conv.conv.out_channels, in_kernel_size=enc.pre_conv.conv.kernel_size[0],
			out_kernel_size=enc.post_conv.conv.kernel_size[0], down_sample_rates=list(enc.down_sample_rates), activation=act(enc.activations[0]),
			**res(enc.res_layers[0])),
		decoder=dict(input_dim=dec.pre_conv.conv.in_channels, base_channels=dec.pre_conv.conv.out_channels, in_kernel_size=dec.pre_conv.conv.kernel_size[0],
			out_kernel_size=dec.post_conv.conv.kernel_size[0], up_sample_rates=list(dec.up_sample_rates), activation=act(dec.activations[0]),
			output_activation=type(dec.out_activation).__name__, **res(dec.res_layers[0])))
	Path(out).parent.mkdir(parents=True, exist_ok=True)
	save_file(weights, str(out), metadata=dict(format="nemo-codec", config=json.dumps(config)))
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


def recommended_guidance(path):
	"""The guidance an export was tuned for ({"cfg", "cfg_prompt"}), or None (older exports, training checkpoints)."""
	if path is None or str(path) == DEFAULT_MODEL:
		path = default_model()
	if Path(path).suffix not in (".sft", ".safetensors"):
		return None
	from safetensors import safe_open
	with safe_open(str(path), framework="pt") as f:
		g = f.metadata().get("guidance")
	return json.loads(g) if g else None


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
	ap.add_argument("ckpt", type=Path, help="a training checkpoint (last.pt), or `codec` for the NeMo codec")
	ap.add_argument("out", type=Path, help="the export (.sft)")
	ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
	ap.add_argument("--cfg", type=float, default=None, help="the guidance the model was tuned for (stored; tts.py's default)")
	ap.add_argument("--cfg-prompt", type=float, default=None, help="the speaker guidance it was tuned for")
	args = ap.parse_args()
	if str(args.ckpt) == "codec":
		export_codec(args.out)
	else:
		export(args.ckpt, args.out, getattr(torch, args.dtype), args.cfg, args.cfg_prompt)
	print(f"wrote {args.out} ({args.out.stat().st_size / 2**20:.0f} MB)")


if __name__ == "__main__":
	main()
