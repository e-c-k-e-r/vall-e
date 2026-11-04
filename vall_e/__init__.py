"""VALL-E v3: text-to-speech over NeMo 44.1 kHz FSQ codes, generated as values by a flow-matching head (docs/).

	python -m vall_e.tts --ckpt <ckpt> --text "Hello." --prompt voice.wav --out out.wav   # inference
	python -m vall_e.train --store <store> --out training/<run> ...                       # training
	python -m vall_e.evalset run --set evalsets/v3-eval.json --ckpt <ckpt> --out <dir>    # evaluation
	python -m vall_e.store convert ...                                                    # data
"""
