#!/bin/bash
# A length head for v3-main5, trained on the frozen model (every addition trained into the backbone cost text use, so
# this one stays out of it): LEN rows only, only len_q / length learn.
# usage: nohup scripts/v3-main5-len.sh >> training/v3-main5-len.log 2>&1 &
cd "$(dirname "$0")/.."
S=data/store
ALL=(Audiobooks LibriTTS-R LibriVox Emilia "GROUP" GROUP GROUP GROUP GROUP GROUP GROUP GROUP GROUP "GROUP" "GROUP"
	"GROUP" GROUP GROUP GROUP "GROUP" GROUP GROUP)
scripts/v3-proto-run.sh training/v3-main5-len --store $S --groups "${ALL[@]}" --min-rate 13 --min-rate-groups LibriVox Audiobooks \
	--init-from training/v3-main5/last.pt --train-only len_q length \
	--head flow --d-model 1024 --layers 16 --heads 16 --batch 32 --max-tokens 32000 \
	--lr 1e-3 --warmup 200 --steps 6000 --decay-frac 0.3 --compile --workers 6 --eval-every 1000 --val-size 256 \
	--len-frac 1.0 --len-ramp 1 --len-weight 1.0 --cond-drop 0 --text-drop 0 --prompt-drop 0
echo "$(date) length head done"
