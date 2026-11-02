#!/bin/bash
# Speaker similarity on the frozen eval set: the ground truth again (now with the prompt clips SIM is measured
# against), the flagship re-scored with SIM, then the inference-only knobs (prompt length, CFG scale, flow
# temperature), each compared with the flagship's default.
# usage: nohup scripts/v3-sim-sweep.sh >> training/v3-sim-sweep.log 2>&1 &
cd "$(dirname "$0")/.."
E=training/v3-evals
P="venv/bin/python -m vall_e.v3.evalset"
F=training/v3-main5p-len/last.pt
[ -f $E/gt/0000_prompt.wav ] || { rm -rf $E/gt; MIOPEN_FIND_MODE=FAST $P run --set evalsets/v3-eval.json --ckpt gt --out $E/gt > $E/gt.log 2>&1; }
$P score --set evalsets/v3-eval.json --out $E/flagship-pred1.0 > $E/flagship-pred1.0.rescore.log 2>&1
run() {  # run <name> <args...>
	local name=$1; shift
	[ -f $E/$name/scores.json ] && return
	echo "$(date) $name"
	MIOPEN_FIND_MODE=FAST $P run --set evalsets/v3-eval.json --out $E/$name --ckpt $F --length predict "$@" > $E/$name.log 2>&1
}
run sim-prompt3 --prompt-seconds 3
run sim-prompt10 --prompt-seconds 10
run sim-cfg1.5 --cfg 1.5
run sim-cfg3.5 --cfg 3.5
run sim-temp0.8 --temperature 0.8
run sim-temp0.6 --temperature 0.6
$P report $E/gt $E/flagship-pred1.0 $E/sim-prompt3 $E/sim-prompt10 $E/sim-cfg1.5 $E/sim-cfg3.5 $E/sim-temp0.8 $E/sim-temp0.6 --gt $E/gt 2>&1 | grep "==\|  all\|  en "
for b in sim-prompt3 sim-prompt10 sim-cfg1.5 sim-cfg3.5 sim-temp0.8 sim-temp0.6; do
	$P compare $E/flagship-pred1.0 $E/$b --gt $E/gt 2>&1 | grep "==\|  all"
done
echo "$(date) sim sweep done"
