#!/bin/bash
# v3-main continued from its step-124k weights with cosine mask ratios and a jointly dropped text + prompt (CFG's null)
# 10% of the time: v3-main (fixed 0.8, independent drops only) barely read the text when sampling. Fresh optimizer
# and schedule over the remaining 276k steps; resumable through v3-proto-run.sh.
# usage: nohup scripts/v3-main2-chain.sh >> training/v3-main2-chain.log 2>&1 &
cd "$(dirname "$0")/.."
S=data/store
out=training/v3-main2
TRAIN_GROUPS=(Audiobooks LibriTTS-R LibriVox Emilia "GROUP" GROUP GROUP GROUP GROUP GROUP GROUP GROUP GROUP "GROUP" "GROUP"
	"GROUP" GROUP GROUP GROUP "GROUP" GROUP GROUP)
[ -f training/v3-main/step-124000-init.pt ] || cp training/v3-main/last.pt training/v3-main/step-124000-init.pt
echo "$(date) training $out"
scripts/v3-proto-run.sh $out --store $S --groups "${TRAIN_GROUPS[@]}" --init-from training/v3-main/step-124000-init.pt \
	--head flow --d-model 1024 --layers 16 --heads 16 --batch 16 --max-tokens 16000 \
	--lr 2e-4 --warmup 500 --steps 276000 --decay-frac 0.2 --compile --workers 6 --keep-every 50000 \
	--mask-ratio cos --cond-drop 0.1 --text-drop 0.05 --prompt-drop 0.1 \
	--ar-frac 0.25 --len-frac 0.05 --len-ramp 1 --accent --prompt-s 3 10 --similar-p 0.75 --noisy-p 0.1 --speaker-alpha 0.5
echo "$(date) training finished"
