# `export.py`

To export the models, run: `python -m vall_e.export --yaml=./training/config.yaml`.

This will export the latest checkpoints, for example, under `./training/ckpt/ar+nar-retnet-8/fp32.pth`, to be loaded on any system with PyTorch, and will include additional metadata, such as the symmap used, and training stats.

Desite being called `fp32.pth`, you can export it to a different precision type with `--dtype=float16|bfloat16|float32`.

You can also export to `safetensors` with `--format=sft`, and `fp32.sft` will be exported instead.