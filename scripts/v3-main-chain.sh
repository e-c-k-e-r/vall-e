#!/bin/bash
# The first full v3 run: bakes noised prompts for 60k random speakers into the store, then trains on the whole store
# with everything the prototype runs found or the plan added (fixed 0.8 masking, AR from step 0, LEN mode, accent
# embedding, separate text / prompt dropout, 3-10 s prompts from the similar lists, noised prompts, speaker-balanced
# draws, capped stop weight). Resumable: the bake is skipped once <store>/noisy exists, and v3-proto-run.sh resumes
# training from last.pt.
# usage: nohup scripts/v3-main-chain.sh >> training/v3-main-chain.log 2>&1 &
cd "$(dirname "$0")/.."
S=data/store
out=training/v3-main

if [ ! -f $S/noisy/source.npy ]; then
	rm -rf $S/noisy.tmp
	echo "$(date) baking noised prompts"
	MIOPEN_FIND_MODE=FAST PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
		venv/bin/python -m vall_e.v3.noisy bake --store $S --speakers 60000 --amp > training/v3-store-noisy.log 2>&1
	rc=$?
	echo "$(date) bake exited with $rc"
	[ $rc -eq 0 ] || exit $rc
fi

echo "$(date) training $out"
scripts/v3-proto-run.sh $out --store $S --head flow --d-model 1024 --layers 16 --heads 16 --batch 16 \
	--lr 2e-4 --warmup 2000 --steps 400000 --decay-frac 0.2 --compile --workers 6 --keep-every 50000 \
	--ar-frac 0.25 --len-frac 0.05 --accent --prompt-s 3 10 --similar-p 0.75 --noisy-p 0.1 --speaker-alpha 0.5
echo "$(date) training finished"
