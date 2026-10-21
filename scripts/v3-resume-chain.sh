#!/bin/bash
# Resumes the queue after the 2026-10-21 OOM: the cosine arm of the mask ablation (from its last checkpoint) and its
# scoring, the AR fine-tune with its scoring, then the prompt-length check on v3-flow-large (on the GPU).
cd "$(dirname "$0")/.."
out=training/v3-flow-maskcos
scripts/v3-proto-run.sh $out --head flow --steps 50000 --batch 32 --compile
venv/bin/python -m vall_e.v3.sample --ckpt $out/last.pt --out $out/samples --n 32 --score >> $out/chain.log 2>&1
echo "$(date) maskcos scored" >> $out/chain.log
scripts/v3-ar-finetune.sh 999999999   # no PID to wait on
S=training/v3-flow-large/promptlen; rm -rf $S; mkdir -p $S
for L in 3 6 10; do
	venv/bin/python -m vall_e.v3.sample --ckpt training/v3-flow-large/last.pt --out $S/p$L --n 32 --score --prompt-seconds $L > $S/p$L.log 2>&1
	echo "prompt ${L}s $(grep -E '^gen ' $S/p$L.log)" >> $S/results.txt
done
