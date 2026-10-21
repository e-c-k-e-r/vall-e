"""Prototype of the `_v3` model (see docs/models_v3.md): NeMo 44KHz audio as ordinal FSQ values instead of tokens.

Self-contained on purpose; it moves into the framework (base_v3.py, ar_nar_v3.py) once it proves viable.

	python -m vall_e.v3.train --out training/v3-proto
	python -m vall_e.v3.sample --ckpt training/v3-proto/last.pt --out training/v3-proto/samples
"""
