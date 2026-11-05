#!/bin/bash
# DPO on the model's own takes (vall_e/posttrain.py pairs + dpo), then the eval set for each kept snapshot against a
# baseline run of the starting model.
# usage: scripts/dpo.sh <init ckpt> <pairs dir> <out dir> <baseline eval dir> [dpo args...]
cd "$(dirname "$0")/.."
. scripts/local.sh
init=$1 pairs=$2 out=$3 base=$4; shift 4
PY=${VALL_E_EVAL_PY:-venv/bin/python}  # it scores (Whisper, WavLM): the eval interpreter
E=training/evals; name=$(basename "$out")
N=${DPO_ITEMS:-4000}
# resumable by chunk; retried, as ROCm segfaulted once 1,500 items in
for try in 1 2 3 4 5; do
	[ $(ls "$pairs"/chunk-*.pt 2>/dev/null | wc -l) -ge $(( (N + 99) / 100 )) ] && break
	$PY -m vall_e.posttrain pairs --init "$init" --out "$pairs" --store $S "${GROUPS_ARGS[@]}" --items $N >> "$pairs.log" 2>&1
done
echo "$(date) pairs: $(ls "$pairs"/chunk-*.pt | wc -l) chunks"
$PY -m vall_e.posttrain dpo --init "$init" --pairs "$pairs" --out "$out" "$@" >> "$out.log" 2>&1
echo "$(date) $name: $(grep '^[0-9]* items\|val_acc' "$out.log" | sed -n '1p;$p' | tr '\n' ' ')"
shopt -s nullglob
for ck in "$out"/step-*.pt; do
	scripts/evaluate.sh "$ck" $E/$name-$(basename "$ck" .pt) "$base" | grep "vs\|  all\|  en "
done
