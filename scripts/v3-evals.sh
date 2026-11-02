#!/bin/bash
# The larger evaluation (evalsets/v3-eval.json, 237 items) over the current models and settings, then a report and
# paired comparisons. Each run is skipped once it has scores.json.
# usage: nohup scripts/v3-evals.sh >> training/v3-evals.log 2>&1 &
cd "$(dirname "$0")/.."
E=training/v3-evals
run() {  # run <name> <evalset.py run args...>
	local name=$1; shift
	[ -f $E/$name/scores.json ] && return
	echo "$(date) $name"
	MIOPEN_FIND_MODE=FAST venv/bin/python -m vall_e.v3.evalset run --set evalsets/v3-eval.json --out $E/$name "$@" > $E/$name.log 2>&1
}
run gt --ckpt gt
run flagship-pred1.0 --ckpt training/v3-main5p-len/last.pt --length predict --length-scale 1.0
run flagship-ref --ckpt training/v3-main5p-len/last.pt --length ref
run pad-pred1.2 --ckpt training/v3-main5p-pad-len/last.pt --length predict --length-scale 1.2
run pad-pred1.0 --ckpt training/v3-main5p-pad-len/last.pt --length predict --length-scale 1.0
run flagship-pred1.2 --ckpt training/v3-main5p-len/last.pt --length predict --length-scale 1.2
P="venv/bin/python -m vall_e.v3.evalset"
$P report $E/gt $E/flagship-ref $E/flagship-pred1.0 $E/flagship-pred1.2 $E/pad-pred1.0 $E/pad-pred1.2 --gt $E/gt 2>&1 | grep -v "amdgpu\|Error while\|dac"
for pair in "flagship-pred1.0 pad-pred1.2" "flagship-pred1.0 pad-pred1.0" "flagship-ref flagship-pred1.0" "flagship-pred1.0 flagship-pred1.2"; do
	set -- $pair; $P compare $E/$1 $E/$2 --gt $E/gt 2>&1 | grep -v "amdgpu\|Error while\|dac"
done
echo "$(date) evals done"
