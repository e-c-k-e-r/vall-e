#!/bin/bash
# Game-voice fine-tune: v1.2 with the game groups drawn as 30% of all draws (they're under 1% of the store) and the
# held-out speakers (data/holdout-speakers.json) never trained on; then weight mixes with v1.2 (scripts/mix.py, length
# heads retrained), scored on the held-out game speakers (evalsets/v3-games-holdout.json) and the main eval set.
cd "$(dirname "$0")/.."
. scripts/local.sh
BASE=${BASE:-training/v3-long-soup0.7-len/last.pt}
O=training/v3-games
E=training/evals; P=${VALL_E_EVAL_PY:-venv/bin/python}
GAMES=(TF2 HL2 HL Portal P3 P4 KH PPP housamo "Deus Ex" "System Shock" SA2 FFVIIR Cyberpunk "Elden Ring" RE4Make "Sam and Max" "Stanley Parable")
MODEL=(--head flow --d-model 1024 --layers 16 --heads 16 --compile --workers 6)
DATA=(--store $S "${GROUPS_ARGS[@]}" --min-rate 13 --min-rate-groups LibriVox Audiobooks --max-s 30 --holdout-speakers data/holdout-speakers.json)
lenhead() {  # <init> <out>
	scripts/train.sh $2 "${DATA[@]}" "${MODEL[@]}" --init-from $1 --train-only len_q length --batch 32 --max-tokens 32000 --lr 1e-3 --warmup 200 \
		--steps 6000 --decay-frac 0.3 --eval-every 1000 --val-size 256 --len-frac 1.0 --len-ramp 1 --len-weight 1.0 --cond-drop 0 --text-drop 0 \
		--prompt-drop 0 --prompt-s 3 10 > /dev/null 2>&1
}
games() {  # <ckpt> <name>
	[ -f $E/holdout-$2/scores.json ] || MIOPEN_FIND_MODE=FAST $P -m vall_e.evalset run --set evalsets/v3-games-holdout.json --ckpt $1 --out $E/holdout-$2 \
		--length predict --cfg 6.0 --cfg-prompt 2.0 --gt $E/holdout-gt > $E/holdout-$2.log 2>&1
}
[ -f $E/holdout-gt/scores.json ] || MIOPEN_FIND_MODE=FAST $P -m vall_e.evalset run --set evalsets/v3-games-holdout.json --ckpt gt --out $E/holdout-gt > $E/holdout-gt.log 2>&1
games data/models/nemo-44khz-nar-flow-225m-v1.2.sft v1.2
echo "$(date) fine-tune"
scripts/train.sh $O "${DATA[@]}" --oversample-groups "${GAMES[@]}" --oversample-p 0.3 "${MODEL[@]}" --batch 16 --max-tokens 16000 --mask-ratio cos \
	--cond-drop 0.1 --text-drop 0 --prompt-drop 0.1 --prompt-s 3 10 --init-from $BASE --lr 5e-5 --warmup 500 --steps 15000 --decay-frac 0.3 --keep-every 5000 > /dev/null 2>&1
echo "$(date) fine-tune done"
for a in 1.0 0.7 0.5; do
	n=games-a$a; ck=training/v3-$n-len/last.pt
	if [ ! -f $ck ]; then
		if [ $a = 1.0 ]; then init=$O/last.pt; else mkdir -p training/v3-games-mix; init=training/v3-games-mix/a$a.pt; [ -f $init ] || $P scripts/mix.py $O/last.pt $BASE $a $init > /dev/null; fi
		lenhead $init training/v3-$n-len
	fi
	games $ck $n
	echo "== $n, held-out game speakers (A: v1.2):"; $P -m vall_e.evalset compare $E/holdout-v1.2 $E/holdout-$n --gt $E/holdout-gt 2>&1 | grep "  all\|  en \|  ja "
	echo "== $n, main set (A: v1.2):"; scripts/evaluate.sh $ck $E/$n $E/long-soup0.7-regular | grep "  all\|  en " | tail -2
done
echo "$(date) games chain done"
