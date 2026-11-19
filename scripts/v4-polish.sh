#!/bin/bash
# v4's polish, as v1.3 got (checkpoint averaging, then the Genshin fine-tune with weight mixes):
#   scripts/v4-polish.sh avg        averages of v4-more's checkpoints, scored with the real lengths (no length heads
#                                   needed) on the main, held-out game, held-out Genshin and Nikke sets
#   AVG=<ckpt> scripts/v4-polish.sh ft    a length head on AVG, the 40k-step Genshin fine-tune from it (games and Genshin
#                                   at 40% of draws), mixes at 0.7 / 0.5 with length heads, all scored against v1.3
cd "$(dirname "$0")/.."
. scripts/local.sh
E=training/evals; P=${VALL_E_EVAL_PY:-venv/bin/python}; G=~/Documents/datasets/training/dataset/44KHz-nemo-genshin
export MIOPEN_FIND_MODE=FAST
CFG=(--cfg 3.0 --cfg-prompt 2.0)  # the sweep's pick for v4 (6.0 / 2.0 was tuned on 225M)
SETS=("main evalsets/v3-eval.json gt" "hgames evalsets/v3-games-holdout.json holdout-gt" "genshin evalsets/v3-genshin-holdout.json genshin-gt" "nikke evalsets/v3-nikke.json nikke-gt")
run() {  # <set> <ckpt> <out> <gt> <length>
	for try in 1 2 3 4; do
		[ -f $3/scores.json ] && return
		$P -m vall_e.evalset run --set $1 --ckpt $2 --out $3 --length $5 "${CFG[@]}" --gt $4 >> $3.log 2>&1
	done
}
OVER=(TF2 HL2 HL Portal P3 P4 KH PPP housamo "Deus Ex" "System Shock" SA2 FFVIIR Cyberpunk "Elden Ring" RE4Make "Sam and Max" "Stanley Parable" Genshin)
MODEL=(--head flow --d-model 1536 --layers 24 --heads 24 --compile --grad-ckpt --workers 6)
DATA=(--store $S "${GROUPS_ARGS[@]}" --extra-store $G --min-rate 13 --min-rate-groups LibriVox Audiobooks --max-s 30 --holdout-speakers data/holdout-speakers-v2.json)
lenhead() {  # <init> <out>
	[ -f $2/last.pt ] || scripts/train.sh $2 "${DATA[@]}" "${MODEL[@]}" --init-from $1 --train-only len_q length --batch 32 --max-tokens 32000 --lr 1e-3 \
		--warmup 200 --steps 6000 --decay-frac 0.3 --eval-every 1000 --val-size 256 --len-frac 1.0 --len-ramp 1 --len-weight 1.0 --cond-drop 0 --text-drop 0 \
		--prompt-drop 0 --prompt-s 3 10 > /dev/null 2>&1
}
case "$1" in
avg)
	A=training/v4-avg; mkdir -p $A
	[ -f $A/avg2.pt ] || $P scripts/avg.py $A/avg2.pt training/v4-more/step-200000.pt training/v4-more/step-300000.pt
	[ -f $A/avg3.pt ] || $P scripts/avg.py $A/avg3.pt training/v4-more/step-100000.pt training/v4-more/step-200000.pt training/v4-more/step-300000.pt
	for m in "training/v4-more/step-300000.pt last" "$A/avg2.pt avg2" "$A/avg3.pt avg3"; do
		set -- $m; ck=$1 n=$2
		for s in "${SETS[@]}"; do set -- $s; run $2 $ck $E/ref-$1-v4$n $E/$3 ref; done
	done
	for s in "${SETS[@]}"; do
		set -- $s
		for n in avg2 avg3; do echo "== $1: v4 last (A) vs $n (B), real lengths"; $P -m vall_e.evalset compare $E/ref-$1-v4last $E/ref-$1-v4$n --gt $E/$3 2>&1 | grep "  all"; done
	done ;;
ft)
	: ${AVG:?the average to polish}
	lenhead $AVG training/v4-avg-len
	O=training/v4-genshin
	[ -f $O/last.pt ] || scripts/train.sh $O "${DATA[@]}" --oversample-groups "${OVER[@]}" --oversample-p 0.4 "${MODEL[@]}" --batch 16 --max-tokens 16000 \
		--mask-ratio cos --cond-drop 0.1 --text-drop 0 --prompt-drop 0.1 --prompt-s 3 10 --init-from training/v4-avg-len/last.pt --lr 3e-5 --warmup 500 \
		--steps 40000 --decay-frac 0.3 --keep-every 20000 > /dev/null 2>&1
	for a in 0.7 0.5; do
		mkdir -p training/v4-genshin-mix; [ -f training/v4-genshin-mix/a$a.pt ] || $P scripts/mix.py $O/last.pt training/v4-avg-len/last.pt $a training/v4-genshin-mix/a$a.pt
		lenhead training/v4-genshin-mix/a$a.pt training/v4-genshin-a$a-len
	done
	for m in "training/v4-avg-len/last.pt v4avg" "training/v4-genshin-a0.7-len/last.pt v4g0.7" "training/v4-genshin-a0.5-len/last.pt v4g0.5"; do
		set -- $m; ck=$1 n=$2
		for s in "${SETS[@]}"; do set -- $s; run $2 $ck $E/pol-$1-$n $E/$3 predict; done
	done
	for s in "${SETS[@]}"; do
		set -- $s
		for n in v4avg v4g0.7 v4g0.5; do echo "== $1: v1.3 at its own 6.0 / 2.0 (A) vs $n at 3.0 / 2.0 (B)"; $P -m vall_e.evalset compare $E/$1-v1.3r $E/pol-$1-$n --gt $E/$3 2>&1 | grep "  all\|  en \|  ja "; done
	done
	echo "$(date) polish done" ;;
*) sed -n 2,6p "$0"; exit 1 ;;
esac
