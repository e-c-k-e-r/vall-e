#!/bin/bash
# Character-voice fine-tune: v1.2 on the main store plus the Genshin store (python -m vall_e.ingest genshin; 513 h of
# en + ja character lines), the game groups and Genshin drawn as 40% of all draws, the held-out speakers
# (data/holdout-speakers-v2.json: the 13 game speakers and 10 Genshin characters in both languages) never trained on;
# then weight mixes with v1.2 (scripts/mix.py, length heads retrained), scored on the held-out Genshin characters
# (evalsets/v3-genshin-holdout.json), the held-out game speakers, both with 3-clip prompts too, and the main eval set.
#   GENSHIN=<store> scripts/genshin-chain.sh          SMOKE=1: only the first 520 fine-tune steps (one log interval), into training/v3-genshin-smoke
cd "$(dirname "$0")/.."
. scripts/local.sh
BASE=${BASE:-training/v3-long-soup0.7-len/last.pt}
G=${GENSHIN:?the Genshin store}
O=training/v3-genshin
E=training/evals; P=${VALL_E_EVAL_PY:-venv/bin/python}
export MIOPEN_FIND_MODE=FAST
OVER=(TF2 HL2 HL Portal P3 P4 KH PPP housamo "Deus Ex" "System Shock" SA2 FFVIIR Cyberpunk "Elden Ring" RE4Make "Sam and Max" "Stanley Parable" Genshin)
MODEL=(--head flow --d-model 1024 --layers 16 --heads 16 --compile --workers 6)
DATA=(--store $S "${GROUPS_ARGS[@]}" --extra-store $G --min-rate 13 --min-rate-groups LibriVox Audiobooks --max-s 30 --holdout-speakers data/holdout-speakers-v2.json)
FT=(--oversample-groups "${OVER[@]}" --oversample-p 0.4 "${MODEL[@]}" --batch 16 --max-tokens 16000 --mask-ratio cos \
	--cond-drop 0.1 --text-drop 0 --prompt-drop 0.1 --prompt-s 3 10 --init-from $BASE --lr 5e-5 --warmup 500 --decay-frac 0.3)
if [ -n "$SMOKE" ]; then
	scripts/train.sh training/v3-genshin-smoke "${DATA[@]}" "${FT[@]}" --steps 520 --eval-every 1000000 --keep-every 1000000
	exit
fi
lenhead() {  # <init> <out>
	scripts/train.sh $2 "${DATA[@]}" "${MODEL[@]}" --init-from $1 --train-only len_q length --batch 32 --max-tokens 32000 --lr 1e-3 --warmup 200 \
		--steps 6000 --decay-frac 0.3 --eval-every 1000 --val-size 256 --len-frac 1.0 --len-ramp 1 --len-weight 1.0 --cond-drop 0 --text-drop 0 \
		--prompt-drop 0 --prompt-s 3 10 > /dev/null 2>&1
}
run() {  # <set> <ckpt> <out> <gt>: retried (ROCm segfaults; a run resumes)
	for try in 1 2 3 4; do
		[ -f $3/scores.json ] && return
		$P -m vall_e.evalset run --set $1 --ckpt $2 --out $3 --length predict --cfg 6.0 --cfg-prompt 2.0 ${4:+--gt $4} >> $3.log 2>&1
	done
}
gt() {  # <set> <out>
	for try in 1 2 3; do [ -f $2/scores.json ] && return; $P -m vall_e.evalset run --set $1 --ckpt gt --out $2 >> $2.log 2>&1; done
}
# "hgames", not "games": $E/games-<model> holds runs on v3-games.json (a different set), which run() would take as done
SETS=("genshin evalsets/v3-genshin-holdout.json $E/genshin-gt" "hgames evalsets/v3-games-holdout.json $E/holdout-gt")
held() {  # <ckpt> <name>: both held-out sets, with their own prompt and 3-clip prompts
	for s in "${SETS[@]}"; do
		set -- $s; n=$1 set=$2 g=$3
		run $set $CK $E/$n-$NAME $g
		run training/prompts-$n-concat3.json $CK $E/$n-c3-$NAME $g
	done
}
# references and prompt variants (select_prompts.py) for the Genshin set; the game set's exist (ps-chain)
gt evalsets/v3-genshin-holdout.json $E/genshin-gt
for try in 1 2 3; do [ -f training/prompts-genshin-concat3.json ] && break; $P training/calibration/select_prompts.py evalsets/v3-genshin-holdout.json training/prompts-genshin > training/prompts-genshin.log 2>&1; done
CK=data/models/nemo-44khz-nar-flow-225m-v1.2.sft NAME=v1.2 held
echo "$(date) baselines done"
[ -f $O/last.pt ] || scripts/train.sh $O "${DATA[@]}" "${FT[@]}" --steps ${STEPS:-40000} --keep-every 10000 > /dev/null 2>&1
[ -f $O/last.pt ] || { echo "$(date) fine-tune FAILED"; exit 1; }
echo "$(date) fine-tune done"
for a in 1.0 0.7 0.5; do
	NAME=genshin-a$a; CK=training/v3-$NAME-len/last.pt
	if [ ! -f $CK ]; then
		if [ $a = 1.0 ]; then init=$O/last.pt; else mkdir -p training/v3-genshin-mix; init=training/v3-genshin-mix/a$a.pt; [ -f $init ] || $P scripts/mix.py $O/last.pt $BASE $a $init > /dev/null; fi
		lenhead $init training/v3-$NAME-len
	fi
	held
	for s in genshin genshin-c3 hgames hgames-c3; do
		g=$E/genshin-gt; [[ $s == hgames* ]] && g=$E/holdout-gt
		echo "== $NAME, held-out $s (A: v1.2):"; $P -m vall_e.evalset compare $E/$s-v1.2 $E/$s-$NAME --gt $g 2>&1 | grep "  all\|  en \|  ja "
	done
	echo "== $NAME, main set (A: v1.2):"; scripts/evaluate.sh $CK $E/$NAME $E/long-soup0.7-regular | grep "  all\|  en " | tail -2
done
echo "$(date) genshin chain done"
