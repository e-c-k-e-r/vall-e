#!/bin/bash
# v3-main5 (started by v3-main5-after-ablate.sh: the R2f recipe at full size): rate-filtered store, NAR only, cosine mask
# ratios, text + prompt dropped together for CFG. Resumes from last.pt through v3-proto-run.sh.
# usage: nohup scripts/v3-main5-chain.sh >> training/v3-main5-chain.log 2>&1 &
cd "$(dirname "$0")/.."
S=data/store
ALL=(Audiobooks LibriTTS-R LibriVox Emilia "GROUP" GROUP GROUP GROUP GROUP GROUP GROUP GROUP GROUP "GROUP" "GROUP"
	"GROUP" GROUP GROUP GROUP "GROUP" GROUP GROUP)
out=training/v3-main5
echo "$(date) training $out"
scripts/v3-proto-run.sh $out --store $S --groups "${ALL[@]}" --min-rate 13 --min-rate-groups LibriVox Audiobooks \
	--head flow --d-model 1024 --layers 16 --heads 16 --batch 16 --max-tokens 16000 \
	--lr 3e-4 --warmup 1000 --steps 300000 --decay-frac 0.2 --compile --workers 6 --keep-every 50000 \
	--mask-ratio cos --cond-drop 0.1 --text-drop 0 --prompt-drop 0
echo "$(date) training finished"
