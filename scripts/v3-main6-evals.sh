#!/bin/bash
# Paired evaluations of v3-main6 along the way: at each step, pause the run, score its checkpoint on the eval set with
# reference lengths (the length head is retrained at the end) at guidance 6.0, with and without speaker guidance, and
# resume. Baseline: the flagship at the same settings.
# usage: nohup scripts/v3-main6-evals.sh >> training/v3-main6-evals.log 2>&1 &
cd "$(dirname "$0")/.."
E=training/v3-evals
P="venv/bin/python -m vall_e.v3.evalset"
out=training/v3-main6
ev() {  # ev <name> <ckpt> <args...>
	local name=$1 ckpt=$2; shift 2
	[ -f $E/$name/scores.json ] || MIOPEN_FIND_MODE=FAST $P run --set evalsets/v3-eval.json --out $E/$name --ckpt $ckpt --length ref --cfg 6.0 "$@" > $E/$name.log 2>&1
}
ev flagship-ref6 training/v3-main5p-len/last.pt
# training starts once the baseline is scored (no sharing the GPU), with the text-use gate (v3-main5p: ~5.8%)
if ! ps -eo args | grep -v grep | grep -q "vall_e.v3.train --out $out"; then
	nohup scripts/v3-main6-chain.sh >> training/v3-main6-chain.log 2>&1 &
	nohup scripts/v3-textuse-gate.sh $out 10000 0.04 >> training/v3-main6-chain.log 2>&1 &
fi
for step in 25000 50000 100000 150000 200000 250000 300000; do
	until grep -q "\"step\": $step}" $out/log.jsonl 2>/dev/null; do sleep 120; done
	sleep 60
	if [ $step -lt 300000 ]; then
		for pat in "v3-main6-chain.sh" "v3-proto-run.sh $out" "vall_e.v3.train --out $out"; do
			ps -eo pid,args | grep -F "$pat" | grep -v grep | awk '{print $1}' | xargs -r kill; sleep 5
		done
		sleep 20; ps -eo pid,comm | awk '$2=="pt_data_worker"{print $1}' | xargs -r kill; sleep 10
	fi
	cp $out/last.pt $out/eval-$step.pt
	echo "$(date) step $step: text_use $(python3 -c "import json; print(round([json.loads(l) for l in open('$out/log.jsonl') if 'val/text_use' in l][-1]['val/text_use'], 4))")"
	ev main6-$step $out/eval-$step.pt
	ev main6-$step-sp2 $out/eval-$step.pt --cfg-prompt 2.0
	rm -f $out/eval-$step.pt
	for b in main6-$step main6-$step-sp2; do $P compare $E/flagship-ref6 $E/$b --gt $E/gt 2>&1 | grep "==\|  all\|  en "; done
	[ $step -lt 300000 ] && nohup scripts/v3-main6-chain.sh >> training/v3-main6-chain.log 2>&1 &
	sleep 5
done
echo "$(date) main6 evals done"
