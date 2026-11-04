#!/bin/bash
# GRPO on the length head (vall_e/posttrain.py length), then the eval set against the starting model.
# usage: scripts/grpo-length.sh <init ckpt> <out dir> [posttrain args...]
cd "$(dirname "$0")/.."
. scripts/local.sh
init=$1 out=$2; shift 2
PY=${VALL_E_EVAL_PY:-venv/bin/python}  # it scores (Whisper, WavLM): the eval interpreter
E=training/evals; name=$(basename "$out")
$PY -m vall_e.posttrain length --init "$init" --out "$out" --store $S "${GROUPS_ARGS[@]}" "$@" >> "$out.log" 2>&1
echo "$(date) $name done: $(grep '^{' "$out.log" | grep probe | tail -1)"
# the baseline at the same code (predict_length's head in fp32)
scripts/evaluate.sh "$init" $E/$name-base > /dev/null
shopt -s nullglob
for ck in "$out"/step-*.pt; do
	scripts/evaluate.sh "$ck" $E/$name-$(basename "$ck" .pt) $E/$name-base | grep "vs\|  all\|  en "
done
