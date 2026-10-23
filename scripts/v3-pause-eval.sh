#!/bin/bash
# Waits for v3-main's step-<N> checkpoint, pauses training, scores that checkpoint on the GPU (per language, NAR and AR,
# plus the new inference options), then resumes training. Each eval is skipped if its scores.json exists, and
# training is resumed whatever the evals do.
# usage: nohup scripts/v3-pause-eval.sh 100000 >> training/v3-main/evals.log 2>&1 &
cd "$(dirname "$0")/.."
step=$1
out=training/v3-main
ckpt=$out/step-$step.pt
ev=$out/eval-$step
size() { stat -c %s "$1" 2>/dev/null || echo 0; }

until [ -f $ckpt ]; do sleep 60; done
# the copy is written in place after last.pt: wait for it to stop growing
while [ "$(size $ckpt)" != "$(size $out/last.pt)" ]; do sleep 10; done
sleep 30
echo "$(date) $ckpt ready, pausing training"
# the chain and the retry wrapper first, so nothing restarts the trainer
for pat in "v3-main-chain.sh" "v3-proto-run.sh training/v3-main" "vall_e.v3.train --out training/v3-main"; do
	ps -eo pid,args | grep -F "$pat" | grep -v grep | awk '{print $1}' | xargs -r kill
	sleep 5
done
sleep 20
ps -eo pid,comm | awk '$2=="pt_data_worker"{print $1}' | xargs -r kill
sleep 10

mkdir -p $ev
run() {  # run <name> <sample.py args...>
	local name=$1; shift
	[ -f $ev/$name/scores.json ] && return
	venv/bin/python -m vall_e.v3.sample --ckpt $ckpt --out $ev/$name --score "$@" > $ev/$name.log 2>&1
	echo "$(date) $name: $(grep -h '^gen' $ev/$name.log)"
}
run nar --per-language 4
run nar-predlen --per-language 4 --length predict
run ar --per-language 4 --mode ar --cfg 1.5
run nar-textnull --per-language 4 --null text
run nar-noaccent --per-language 4 --lang unknown
run nar-promptless --per-language 2 --promptless
echo "$(date) evals done, resuming training"
nohup scripts/v3-main-chain.sh >> training/v3-main-chain.log 2>&1 &
