#!/bin/bash
# Silence padding against over-duration: the over-duration test on the current flagship (v3-main5p-len), a fine-tune
# with 30% of targets padded with trailing silence (stopped if val/text_use < 4% at 10k), a fresh length head on the
# frozen result (trained on unpadded lengths), and the same test on that.
# usage: nohup scripts/v3-pad-chain.sh >> training/v3-pad-chain.log 2>&1 &
cd "$(dirname "$0")/.."
S=data/store
ALL=(Audiobooks LibriTTS-R LibriVox Emilia "GROUP" GROUP GROUP GROUP GROUP GROUP GROUP GROUP GROUP "GROUP" "GROUP"
	"GROUP" GROUP GROUP GROUP "GROUP" GROUP GROUP)
DATA=(--store $S --groups "${ALL[@]}" --min-rate 13 --min-rate-groups LibriVox Audiobooks)
MODEL=(--head flow --d-model 1024 --layers 16 --heads 16 --compile --workers 6)
overdur() {  # overdur <name> <ckpt>
	[ -f training/v3-pad-$1-overdur.log ] && return
	venv/bin/python scratch/v3_overdur.py $2 training/v3-pad-overdur/$1 2>&1 | grep "^#\|Error" > training/v3-pad-$1-overdur.log
	echo "$(date) over-duration, $1: $(tr '\n' ' ' < training/v3-pad-$1-overdur.log)"
}
overdur before training/v3-main5p-len/last.pt

out=training/v3-main5p-pad
nohup scripts/v3-textuse-gate.sh $out 10000 0.04 >> training/v3-pad-chain.log 2>&1 &
scripts/v3-proto-run.sh $out "${DATA[@]}" "${MODEL[@]}" --init-from training/v3-main5p-len/last.pt \
	--batch 16 --max-tokens 16000 --lr 1e-4 --warmup 500 --steps 20000 --decay-frac 0.3 \
	--mask-ratio cos --cond-drop 0.1 --text-drop 0 --prompt-drop 0 --prompt-s 3 10 --pad-p 0.3
step=$(grep -o '"step": [0-9]*}' $out/log.jsonl | tail -1 | grep -o '[0-9]*')
echo "$(date) pad fine-tune ended at step $step"
[ "$step" = "20000" ] || exit 0

scripts/v3-proto-run.sh training/v3-main5p-pad-len "${DATA[@]}" "${MODEL[@]}" --init-from $out/last.pt --train-only len_q length \
	--batch 32 --max-tokens 32000 --lr 1e-3 --warmup 200 --steps 6000 --decay-frac 0.3 --eval-every 1000 --val-size 256 \
	--len-frac 1.0 --len-ramp 1 --len-weight 1.0 --cond-drop 0 --text-drop 0 --prompt-drop 0 --prompt-s 3 10
echo "$(date) length head: $(grep '\[eval\]' training/v3-main5p-pad-len/run.log | tail -1 | grep -o 'val@len/rel_err [0-9.]*\|val/text_use [0-9.]*' | tr '\n' ' ')"
overdur after training/v3-main5p-pad-len/last.pt
echo "$(date) pad chain done"
