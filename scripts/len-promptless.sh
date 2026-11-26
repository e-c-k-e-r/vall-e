#!/bin/bash
# Length heads that know promptless inputs: every head so far was trained with --prompt-drop 0, and without a prompt they predict far too
# long (median predicted / real length: v2.0 1.44, v2.0-fast 1.30, v1.3 1.85, against 1.08-1.12 with one), which the model fills with
# extra words. Each head is trained on from its current one with 20% of prompts dropped (backbone frozen), exported as a -lenfix
# candidate with the same guidance, and checked: the length ratios with and without a prompt, and the main set against the current export.
#   scripts/len-promptless.sh
cd "$(dirname "$0")/.."
. scripts/local.sh
E=training/evals; P=${VALL_E_EVAL_PY:-venv/bin/python}; G=~/Documents/datasets/training/dataset/44KHz-nemo-genshin; M=data/models
export MIOPEN_FIND_MODE=FAST
OVER=(TF2 HL2 HL Portal P3 P4 KH PPP housamo "Deus Ex" "System Shock" SA2 FFVIIR Cyberpunk "Elden Ring" RE4Make "Sam and Max" "Stanley Parable" Genshin)
DATA=(--store $S "${GROUPS_ARGS[@]}" --extra-store $G --min-rate 13 --min-rate-groups LibriVox Audiobooks --max-s 30
	--holdout-speakers data/holdout-speakers-v2.json --oversample-groups "${OVER[@]}" --oversample-p 0.15)
BIG=(--head flow --d-model 1536 --layers 24 --heads 24 --compile --grad-ckpt --workers 6)
SMALL=(--head flow --d-model 1024 --layers 16 --heads 16 --compile --workers 6)
LEN=(--train-only len_q length --batch 32 --max-tokens 32000 --lr 3e-4 --warmup 100 --steps 3000 --decay-frac 0.3 --eval-every 1000 --val-size 256
	--len-frac 1.0 --len-ramp 1 --len-weight 1.0 --cond-drop 0 --text-drop 0 --prompt-drop 0.2 --prompt-s 3 10)
# name, current head's checkpoint, model args, guidance (export flags), the current export
for m in "v2.0 training/v4-avg-len/last.pt BIG --cfg 3.0 --cfg-prompt 2.0 nemo-44khz-nar-flow-746m-v2.0" \
	"v2.0-fast training/v4-distill-len/last.pt BIG --cfg 0.0 --cfg-prompt 0.0 nemo-44khz-nar-flow-746m-v2.0-fast" \
	"v1.3 training/v3-genshin-a0.5-len/last.pt SMALL - - - - nemo-44khz-nar-flow-225m-v1.3"; do
	set -- $m
	n=$1 init=$2 O=training/lenfix-$1 X=$M/$8-lenfix.sft
	[ "$3" = BIG ] && MODEL=("${BIG[@]}") || MODEL=("${SMALL[@]}")
	[ "$4" = - ] && GUIDE=() || GUIDE=($4 $5 $6 $7)
	[ "$(grep -o '"step": [0-9]*' $O/log.jsonl 2>/dev/null | tail -1)" = '"step": 3000' ] || \
		scripts/train.sh $O "${DATA[@]}" "${MODEL[@]}" --init-from $init "${LEN[@]}" > /dev/null 2>&1
	[ "$(grep -o '"step": [0-9]*' $O/log.jsonl | tail -1)" = '"step": 3000' ] || { echo "$(date) $n FAILED"; continue; }
	[ -f $X ] || $P -m vall_e.export $O/last.pt $X "${GUIDE[@]}"
	echo "$(date) $n: $(grep 'val@len' $O/log.jsonl | tail -1 | grep -o '"val@len/rel_err": [0-9.]*')"
done
# the length ratios, current against -lenfix
$P - <<'EOF'
import json, torch, numpy as np
from vall_e.tts import TTS
from vall_e.model import predict_length
items = [it for it in json.load(open("evalsets/v3-eval.json"))["items"] if it["language"] == "en"][:60]
for base in ["nemo-44khz-nar-flow-746m-v2.0", "nemo-44khz-nar-flow-746m-v2.0-fast", "nemo-44khz-nar-flow-225m-v1.3"]:
	for name in [base, base + "-lenfix"]:
		tts = TTS(f"data/models/{name}.sft")
		rp, rn = [], []
		for it in items:
			t = tts.phonemes(it["text"], "en"); p = tts.encode_prompt(f"training/evals/gt/{it['id']:04d}_prompt.wav")
			with torch.autocast("cuda", dtype=torch.bfloat16):
				rp.append(predict_length(tts.model, t, p) / it["frames"]); rn.append(predict_length(tts.model, t, p[:0]) / it["frames"])
		q = lambda r: f"{np.median(r):.2f} ({np.percentile(r, 25):.2f}-{np.percentile(r, 75):.2f})"
		print(f"{name}: predicted / real, with prompt {q(rp)}, promptless {q(rn)}", flush=True)
		del tts; torch.cuda.empty_cache()
EOF
# prompted generation unchanged: the main set, current export (A) against -lenfix (B)
for m in "v2.0 nemo-44khz-nar-flow-746m-v2.0 3.0 2.0" "v2.0-fast nemo-44khz-nar-flow-746m-v2.0-fast 0 0"; do
	set -- $m
	for v in $2 $2-lenfix; do
		for t in 1 2 3; do [ -f $E/lf-main-$v/scores.json ] && break; $P -m vall_e.evalset run --set evalsets/v3-eval.json --ckpt $M/$v.sft --out $E/lf-main-$v --length predict --cfg $3 --cfg-prompt $4 --gt $E/gt >> $E/lf-main-$v.log 2>&1; done
	done
	echo "== main: $1 (A) vs $1-lenfix (B)"; $P -m vall_e.evalset compare $E/lf-main-$2 $E/lf-main-$2-lenfix --gt $E/gt 2>&1 | grep "  all\|  en "
done
echo "$(date) lenfix done"
