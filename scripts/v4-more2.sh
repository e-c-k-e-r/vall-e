#!/bin/bash
# v4's training continued: another 300k steps of the long-form stage's settings from where it ended (more passes over
# seen data: on 225M the second 300k stage gave +0.038 SIM), checkpoints every 50k for averaging; then the mean of the
# last three (200k / 250k / 300k), a length head, and the five eval sets against the averaged v4.
cd "$(dirname "$0")/.."
. scripts/local.sh
E=training/evals; P=${VALL_E_EVAL_PY:-venv/bin/python}; G=~/Documents/datasets/training/dataset/44KHz-nemo-genshin; O=training/v4-more2
export MIOPEN_FIND_MODE=FAST
OVER=(TF2 HL2 HL Portal P3 P4 KH PPP housamo "Deus Ex" "System Shock" SA2 FFVIIR Cyberpunk "Elden Ring" RE4Make "Sam and Max" "Stanley Parable" Genshin)
MODEL=(--head flow --d-model 1536 --layers 24 --heads 24 --compile --grad-ckpt --workers 6)
DATA=(--store $S "${GROUPS_ARGS[@]}" --extra-store $G --min-rate 13 --min-rate-groups LibriVox Audiobooks
	--holdout-speakers data/holdout-speakers-v2.json --oversample-groups "${OVER[@]}" --oversample-p 0.15)
[ "$(grep -o '"step": [0-9]*' $O/log.jsonl 2>/dev/null | tail -1)" = '"step": 300000' ] || scripts/train.sh $O "${DATA[@]}" "${MODEL[@]}" \
	--batch 16 --max-tokens 16000 --mask-ratio cos --cond-drop 0.1 --text-drop 0 --prompt-drop 0.1 --prompt-s 3 10 --max-s 30 \
	--init-from training/v4-more/last.pt --lr 7e-5 --warmup 1000 --steps 300000 --decay-frac 0.2 --keep-every 50000 > /dev/null 2>&1
[ "$(grep -o '"step": [0-9]*' $O/log.jsonl | tail -1)" = '"step": 300000' ] || { echo "$(date) more2 FAILED"; exit 1; }
echo "$(date) more2 done"
[ -f $O/avg3.pt ] || $P scripts/avg.py $O/avg3.pt $O/step-200000.pt $O/step-250000.pt $O/step-300000.pt
[ -f $O-len/last.pt ] || scripts/train.sh $O-len "${DATA[@]}" "${MODEL[@]}" --max-s 30 --init-from $O/avg3.pt --train-only len_q length --batch 32 \
	--max-tokens 32000 --lr 1e-3 --warmup 200 --steps 6000 --decay-frac 0.3 --eval-every 1000 --val-size 256 --len-frac 1.0 --len-ramp 1 --len-weight 1.0 \
	--cond-drop 0 --text-drop 0 --prompt-drop 0 --prompt-s 3 10 > /dev/null 2>&1
for s in "main evalsets/v3-eval.json gt" "long evalsets/v3-long.json long-gt" "hgames evalsets/v3-games-holdout.json holdout-gt" "genshin evalsets/v3-genshin-holdout.json genshin-gt" "nikke evalsets/v3-nikke.json nikke-gt"; do
	set -- $s
	for t in 1 2 3 4; do [ -f $E/m2-$1-v4/scores.json ] && break; $P -m vall_e.evalset run --set $2 --ckpt $O-len/last.pt --out $E/m2-$1-v4 --length predict --cfg 3.0 --cfg-prompt 2.0 --gt $E/$3 >> $E/m2-$1-v4.log 2>&1; done
	echo "== $1: averaged v4 (A) vs averaged v4 + 300k more (B)"; $P -m vall_e.evalset compare $E/pol-$1-v4avg $E/m2-$1-v4 --gt $E/$3 2>&1 | grep "  all\|  en \|  ja "
done
echo "$(date) more2 chain done"
