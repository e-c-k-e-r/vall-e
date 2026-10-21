#!/bin/bash
# AR fine-tune of v3-flow-large (40% of samples in AR mode), after PID $1 (the mask ablation) exits, then scoring in
# three modes: NAR demask with backbone CFG, NAR demask with head-only CFG, and streaming-order AR.
cd "$(dirname "$0")/.."
while kill -0 "$1" 2>/dev/null; do sleep 60; done
out=training/v3-flow-large-ar
mkdir -p $out && cp -n training/v3-flow-large/index.json $out/
scripts/v3-proto-run.sh $out --head flow --steps 30000 --batch 16 --lr 1e-4 --warmup 500 --d-model 1024 --layers 16 --heads 16 \
	--compile --ar-frac 0.4 --init-from training/v3-flow-large/last.pt
for m in "nar-backbone:--cfg 2.5 --cfg-mode backbone" "nar-head:--cfg 2.5 --cfg-mode head" "ar:--mode ar --cfg 1.5"; do
	name=${m%%:*}; args=${m#*:}
	venv/bin/python -m vall_e.v3.sample --ckpt $out/last.pt --out $out/samples-$name --n 32 --score $args >> $out/chain-$name.log 2>&1
	echo "$(date) $name scored" >> $out/chain.log
done
