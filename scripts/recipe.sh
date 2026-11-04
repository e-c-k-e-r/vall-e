#!/bin/bash
# How the flagship (training/v3-main6-len) was trained, as runnable stages. Each stage starts from the previous one's
# checkpoint and is resumable (scripts/train.sh). About 2.5 days on one RX 7900 XTX in all.
#   scripts/recipe.sh base      300k steps from scratch: masked generation, cosine mask ratios, text + prompt dropped
#                               together for CFG, the rate-filtered store                     (~26 h; was v3-main5)
#   scripts/recipe.sh prompts   30k steps on 3-10 s prompts (longer prompts stop looping)     (~2.5 h; v3-main5-prompt10)
#   scripts/recipe.sh more      300k more steps at lr 1e-4 with prompt-only dropout added, so speaker guidance works
#                               (~0.7 more epochs: +0.038 SIM, -0.8 WER)                      (~27 h; v3-main6)
#   scripts/recipe.sh length    a length head on the frozen final model                       (~25 min; v3-main6-len)
#   scripts/recipe.sh eval      the eval set at the defaults (guidance 6.0, speaker guidance 2.0, predicted length)
# Lessons behind these choices (docs/train.md): train text reading first and add tasks afterwards (every addition
# trained in from scratch cost text use, and together they stopped it); judge a run by sampled WER / val/text_use,
# never by teacher-forced loss; judge a WSD run at its end or with a decay probe, not on its plateau.
cd "$(dirname "$0")/.."
S=${VALL_E_STORE:-data/store}
GROUPS_=(Audiobooks LibriTTS-R LibriVox Emilia "GROUP" GROUP GROUP GROUP GROUP GROUP GROUP GROUP GROUP "GROUP" "GROUP"
	"GROUP" GROUP GROUP GROUP "GROUP" GROUP GROUP)  # every group but the old Validation hold-out
DATA=(--store $S --groups "${GROUPS_[@]}" --min-rate 13 --min-rate-groups LibriVox Audiobooks)
MODEL=(--head flow --d-model 1024 --layers 16 --heads 16 --compile --workers 6)
NAR=(--batch 16 --max-tokens 16000 --mask-ratio cos --cond-drop 0.1 --text-drop 0)
case "$1" in
base)
	nohup scripts/textuse-gate.sh training/base 30000 0.015 >> training/base-gate.log 2>&1 &
	scripts/train.sh training/base "${DATA[@]}" "${MODEL[@]}" "${NAR[@]}" --prompt-drop 0 \
		--lr 3e-4 --warmup 1000 --steps 300000 --decay-frac 0.2 --keep-every 50000 ;;
prompts)
	scripts/train.sh training/prompts "${DATA[@]}" "${MODEL[@]}" "${NAR[@]}" --prompt-drop 0 --prompt-s 3 10 \
		--init-from training/base/last.pt --lr 1e-4 --warmup 500 --steps 30000 --decay-frac 0.3 ;;
more)
	nohup scripts/textuse-gate.sh training/more 10000 0.04 >> training/more-gate.log 2>&1 &
	scripts/train.sh training/more "${DATA[@]}" "${MODEL[@]}" "${NAR[@]}" --prompt-drop 0.1 --prompt-s 3 10 \
		--init-from training/prompts/last.pt --lr 1e-4 --warmup 1000 --steps 300000 --decay-frac 0.2 --keep-every 50000 ;;
length)
	scripts/train.sh training/flagship "${DATA[@]}" "${MODEL[@]}" --init-from training/more/last.pt --train-only len_q length \
		--batch 32 --max-tokens 32000 --lr 1e-3 --warmup 200 --steps 6000 --decay-frac 0.3 --eval-every 1000 --val-size 256 \
		--len-frac 1.0 --len-ramp 1 --len-weight 1.0 --cond-drop 0 --text-drop 0 --prompt-drop 0 --prompt-s 3 10 ;;
eval)
	scripts/evaluate.sh ${2:-training/flagship/last.pt} training/evals/${3:-flagship} ;;
*)
	sed -n '2,15p' "$0"; exit 1 ;;
esac
