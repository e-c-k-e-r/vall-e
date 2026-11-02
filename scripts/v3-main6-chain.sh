#!/bin/bash
# v3-main6: the v3 flagship (v3-main5p-len) trained a further ~epoch (300k steps; v3-main5 saw ~0.68 of an epoch) on the
# filtered store, now including the recovered Emilia/DE, with prompt-only dropout added so speaker guidance
# (--cfg-prompt) works. lr 1e-4 to build on the model rather than retrain it. Resumable.
# usage: nohup scripts/v3-main6-chain.sh >> training/v3-main6-chain.log 2>&1 &
cd "$(dirname "$0")/.."
S=data/store
ALL=(Audiobooks LibriTTS-R LibriVox Emilia "GROUP" GROUP GROUP GROUP GROUP GROUP GROUP GROUP GROUP "GROUP" "GROUP"
	"GROUP" GROUP GROUP GROUP "GROUP" GROUP GROUP)
out=training/v3-main6
echo "$(date) training $out"
scripts/v3-proto-run.sh $out --store $S --groups "${ALL[@]}" --min-rate 13 --min-rate-groups LibriVox Audiobooks \
	--init-from training/v3-main5p-len/last.pt \
	--head flow --d-model 1024 --layers 16 --heads 16 --batch 16 --max-tokens 16000 \
	--lr 1e-4 --warmup 1000 --steps 300000 --decay-frac 0.2 --compile --workers 6 --keep-every 50000 \
	--mask-ratio cos --cond-drop 0.1 --prompt-drop 0.1 --text-drop 0 --prompt-s 3 10
echo "$(date) training finished"
