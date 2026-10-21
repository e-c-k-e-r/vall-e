#!/bin/bash
# Mask-ratio ablation on the v3 prototype (flow head, 41M, 50k steps, bucketed batches, compiled): a fixed 0.8 ratio
# (AutoNAT, arXiv 2406.05478) vs the per-sample cosine draw. Waits for PID $1 (the previous chain) first.
cd "$(dirname "$0")/.."
while kill -0 "$1" 2>/dev/null; do sleep 60; done
for arm in "mask0.8:--mask-ratio 0.8" "maskcos:"; do
	name=${arm%%:*}; extra=${arm#*:}; out=training/v3-flow-$name
	mkdir -p $out && cp -n training/v3-flow-large/index.json $out/
	scripts/v3-proto-run.sh $out --head flow --steps 50000 --batch 32 --compile $extra
	venv/bin/python -m vall_e.v3.sample --ckpt $out/last.pt --out $out/samples --n 32 --score >> $out/chain.log 2>&1
	echo "$(date) $name scored" >> $out/chain.log
done
