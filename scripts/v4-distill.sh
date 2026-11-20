#!/bin/bash
# Guidance distillation for v4 (as scripts/distill-chain.sh for 225M): the student, initialised from the averaged v4,
# learns its guided velocity at v4's guidance (3.0, speaker guidance 2.0) so it samples with one backbone pass per step
# instead of three; v4's data (main + Genshin, games and Genshin at 15%, held-out speakers excluded). Then a length
# head, and the five eval sets without guidance against v4 at 3.0 / 2.0.
#   scripts/v4-distill.sh          SMOKE=1: one log interval into training/v4-distill-smoke
cd "$(dirname "$0")/.."
. scripts/local.sh
T=training/v4-avg-len/last.pt; O=training/v4-distill
E=training/evals; P=${VALL_E_EVAL_PY:-venv/bin/python}; G=~/Documents/datasets/training/dataset/44KHz-nemo-genshin
export MIOPEN_FIND_MODE=FAST
OVER=(TF2 HL2 HL Portal P3 P4 KH PPP housamo "Deus Ex" "System Shock" SA2 FFVIIR Cyberpunk "Elden Ring" RE4Make "Sam and Max" "Stanley Parable" Genshin)
MODEL=(--head flow --d-model 1536 --layers 24 --heads 24 --compile --grad-ckpt --workers 6)
DATA=(--store $S "${GROUPS_ARGS[@]}" --extra-store $G --min-rate 13 --min-rate-groups LibriVox Audiobooks --max-s 30
	--holdout-speakers data/holdout-speakers-v2.json --oversample-groups "${OVER[@]}" --oversample-p 0.15)
DISTILL=(--distill-from $T --distill-cfg 3.0 --distill-cfg-prompt 2.0 --init-from $T --batch 16 --max-tokens 16000 --mask-ratio cos
	--prompt-s 3 10 --lr 2e-5 --warmup 500 --decay-frac 0.3)
if [ -n "$SMOKE" ]; then
	scripts/train.sh training/v4-distill-smoke "${DATA[@]}" "${MODEL[@]}" "${DISTILL[@]}" --steps 520 --eval-every 1000000
	exit
fi
[ -f $O/last.pt ] && [ "$(grep -o '"step": [0-9]*' $O/log.jsonl | tail -1)" = '"step": 20000' ] || \
	scripts/train.sh $O "${DATA[@]}" "${MODEL[@]}" "${DISTILL[@]}" --steps 20000 --keep-every 10000 > /dev/null 2>&1
echo "$(date) distillation done: $(grep '"loss"' $O/log.jsonl | tail -1 | cut -c1-120)"
[ -f $O-len/last.pt ] || scripts/train.sh $O-len "${DATA[@]}" "${MODEL[@]}" --init-from $O/last.pt --train-only len_q length \
	--batch 32 --max-tokens 32000 --lr 1e-3 --warmup 200 --steps 6000 --decay-frac 0.3 --eval-every 1000 --val-size 256 \
	--len-frac 1.0 --len-ramp 1 --len-weight 1.0 --cond-drop 0 --text-drop 0 --prompt-drop 0 --prompt-s 3 10 > /dev/null 2>&1
echo "$(date) length head done"
for s in "main evalsets/v3-eval.json gt" "long evalsets/v3-long.json long-gt" "hgames evalsets/v3-games-holdout.json holdout-gt" "genshin evalsets/v3-genshin-holdout.json genshin-gt" "nikke evalsets/v3-nikke.json nikke-gt"; do
	set -- $s
	for t in 1 2 3 4; do [ -f $E/dis-$1-v4/scores.json ] && break; $P -m vall_e.evalset run --set $2 --ckpt $O-len/last.pt --out $E/dis-$1-v4 --length predict --cfg 0 --cfg-prompt 0 --gt $E/$3 >> $E/dis-$1-v4.log 2>&1; done
	echo "== $1: v4 at 3.0 / 2.0 (A) vs distilled v4, no guidance (B)"; $P -m vall_e.evalset compare $E/pol-$1-v4avg $E/dis-$1-v4 --gt $E/$3 2>&1 | grep "  all\|  en \|  ja "
done
echo "$(date) v4 distill done"
