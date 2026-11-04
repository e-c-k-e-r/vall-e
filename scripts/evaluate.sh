#!/bin/bash
# Scores a checkpoint on the frozen eval set (evalsets/v3-eval.json) at the inference defaults, and compares it with a
# baseline run if given. The ground-truth run (the ceiling, and the references SIM is measured against) is made once.
# usage: scripts/evaluate.sh <ckpt> <out dir> [baseline run dir] [extra evalset.py run args...]
# e.g.   scripts/evaluate.sh training/flagship/last.pt training/evals/flagship
#        scripts/evaluate.sh training/new/last.pt training/evals/new training/evals/flagship
cd "$(dirname "$0")/.."
ckpt=$1 out=$2 base=$3; shift 3 2>/dev/null || shift $#
PY=${VALL_E_EVAL_PY:-venv/bin/python}
E=$(dirname "$out"); GT=$E/gt
mkdir -p "$E"
[ -f $GT/scores.json ] || MIOPEN_FIND_MODE=FAST $PY -m vall_e.evalset run --set evalsets/v3-eval.json --ckpt gt --out $GT > $GT.log 2>&1
[ -f $out/scores.json ] || MIOPEN_FIND_MODE=FAST $PY -m vall_e.evalset run --set evalsets/v3-eval.json --ckpt "$ckpt" --out "$out" \
	--length predict --cfg 6.0 --cfg-prompt 2.0 --gt $GT "$@" > "$out.log" 2>&1
$PY -m vall_e.evalset report $GT "$out" --gt $GT 2>&1 | grep "==\|  [a-z]"
[ -n "$base" ] && $PY -m vall_e.evalset compare "$base" "$out" --gt $GT 2>&1 | grep "==\|  [a-z]"
