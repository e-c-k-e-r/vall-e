#!/bin/bash
# The first full v3 run, restarted from scratch on the prototype's proven recipe (v3-flow-large: NAR only, cosine mask
# ratios, text + prompt dropped together for CFG) plus the additions that showed no harm: the full store with
# group-preserving speaker balance, deduplicated similar prompts, noised prompts, accent embedding, LEN mode, 3-10 s
# prompts. v3-main..main3 (fixed 0.8 masking from scratch, store-wide speaker balance, AR from step 0) learned the audio
# but barely read the text when sampling. AR becomes a separate fine-tune stage afterwards. Resumable.
# usage: nohup scripts/v3-main4-chain.sh >> training/v3-main4-chain.log 2>&1 &
cd "$(dirname "$0")/.."
S=data/store
out=training/v3-main4
TRAIN_GROUPS=(Audiobooks LibriTTS-R LibriVox Emilia "GROUP" GROUP GROUP GROUP GROUP GROUP GROUP GROUP GROUP "GROUP" "GROUP"
	"GROUP" GROUP GROUP GROUP "GROUP" GROUP GROUP)
echo "$(date) training $out"
scripts/v3-proto-run.sh $out --store $S --groups "${TRAIN_GROUPS[@]}" \
	--head flow --d-model 1024 --layers 16 --heads 16 --batch 16 --max-tokens 16000 \
	--lr 3e-4 --warmup 1000 --steps 300000 --decay-frac 0.2 --compile --workers 6 --keep-every 50000 \
	--mask-ratio cos --cond-drop 0.1 --text-drop 0.05 --prompt-drop 0.1 \
	--len-frac 0.05 --accent --prompt-s 3 10 --similar-p 0.75 --noisy-p 0.1 --speaker-alpha 0.5
echo "$(date) training finished"
