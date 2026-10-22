#!/bin/bash
# After PID $1: the user's note that random mask ratios hurt generation at fewer demask steps. Scores the fixed-0.8
# and cosine-draw ablation models at 8 and 16 steps (25 steps already scored in their chain logs).
cd "$(dirname "$0")/.."
while kill -0 "$1" 2>/dev/null; do sleep 60; done
for arm in mask0.8 maskcos; do for st in 8 16; do
	dir=training/v3-flow-$arm/steps$st
	[ -s $dir/scores.json ] && continue
	venv/bin/python -m vall_e.v3.sample --ckpt training/v3-flow-$arm/last.pt --out $dir --n 32 --score --steps $st > $dir.log 2>&1
	echo "$(date) $arm steps $st: $(grep -E '^gen ' $dir.log)"
done; done
