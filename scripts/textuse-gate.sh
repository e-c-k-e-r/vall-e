#!/bin/bash
# usage: scripts/textuse-gate.sh <run dir> <step> <min text_use>: stops the run if its val/text_use at <step> is below the
# minimum (it isn't reading the text). Scale <step> by batch size: the ablation ladder's batch-32 rungs reached about
# 2% by 10k steps on the full store, so a batch-16 run needs about 20k.
cd "$(dirname "$0")/.."
out=$1; step=$2; min=$3
until grep -q "\"step\": $step}" $out/log.jsonl 2>/dev/null; do sleep 120; done
use=$(python3 -c "import json; print([r for r in map(json.loads, open('$out/log.jsonl')) if r.get('step') == $step and 'val/text_use' in r][-1]['val/text_use'])")
if python3 -c "import sys; sys.exit(0 if float('$use') < $min else 1)"; then
	echo "$(date) $out text_use $use < $min at step $step: stopping it"
	for pat in "scripts/train.sh $out" "vall_e.train --out $out"; do
		ps -eo pid,args | grep -F "$pat" | grep -v grep | awk '{print $1}' | xargs -r kill; sleep 5
	done
	sleep 20; ps -eo pid,comm | awk '$2=="pt_data_worker"{print $1}' | xargs -r kill
else
	echo "$(date) $out text_use $use at step $step: reading the text, continuing"
fi
