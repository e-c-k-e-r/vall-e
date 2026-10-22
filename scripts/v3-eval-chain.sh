#!/bin/bash
# Evaluations of the v3 large and AR runs (all GPU inference). Steps whose scores already exist are skipped, so after a
# crash the chain can simply be restarted: the AR fine-tune in three modes, the prompt-length check on v3-flow-large,
# the shipped v6 model on v3's held-out set, and the mode-embedding swap test.
cd "$(dirname "$0")/.."
run() {  # run <out dir> <sample.py args...>: skipped if <out dir>/scores.json exists
	local dir=$1; shift
	[ -s "$dir/scores.json" ] && { echo "$(date) skip $dir (scored)"; return; }
	venv/bin/python -m vall_e.v3.sample --out "$dir" --n 32 --score "$@" > "$dir.log" 2>&1
	echo "$(date) $dir: $(grep -E '^gen ' "$dir.log")"
}
ar=training/v3-flow-large-ar/last.pt
run training/v3-flow-large-ar/samples-nar-backbone --ckpt $ar --cfg 2.5 --cfg-mode backbone
run training/v3-flow-large-ar/samples-nar-head --ckpt $ar --cfg 2.5 --cfg-mode head
run training/v3-flow-large-ar/samples-ar --ckpt $ar --mode ar --cfg 1.5
for L in 3 6 10; do
	run training/v3-flow-large/promptlen/p$L --ckpt training/v3-flow-large/last.pt --prompt-seconds $L
done
if [ ! -s training/v3-flow-large/v6-compare/scores.json ]; then
	venv/bin/python scratch/vallef/v6_on_v3val.py > training/v3-flow-large/v6-compare.log 2>&1
	echo "$(date) v6: $(grep -E '^v6 ' training/v3-flow-large/v6-compare.log)"
fi
run training/v3-flow-large-ar/samples-nar-backbone-swapped --ckpt $ar --cfg 2.5 --cfg-mode backbone --swap-mode-embedding
run training/v3-flow-large-ar/samples-ar-swapped --ckpt $ar --mode ar --cfg 1.5 --swap-mode-embedding
echo "$(date) eval chain done"
