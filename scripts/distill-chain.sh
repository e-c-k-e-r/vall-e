#!/bin/bash
# Guidance distillation: a student (initialised from the teacher) learns the teacher's guided velocity (cfg 6, speaker
# guidance 2) so it samples without guidance, one backbone pass per step instead of three. Then a length head, and both
# eval sets at cfg 0 against the teacher at its defaults.
cd "$(dirname "$0")/.."
. scripts/local.sh
T=${TEACHER:-training/v3-long-soup0.7-len/last.pt}
O=${OUT:-training/v3-distill}
E=training/evals; P=${VALL_E_EVAL_PY:-venv/bin/python}
MODEL=(--head flow --d-model 1024 --layers 16 --heads 16 --compile --workers 6)
DATA=(--store $S "${GROUPS_ARGS[@]}" --min-rate 13 --min-rate-groups LibriVox Audiobooks --max-s 30 --long-p 0.2)
scripts/train.sh $O "${DATA[@]}" "${MODEL[@]}" --distill-from $T --init-from $T --batch 16 --max-tokens 16000 --mask-ratio cos \
	--prompt-s 3 10 --lr 3e-5 --warmup 500 --steps ${STEPS:-20000} --decay-frac 0.3 --keep-every 10000
echo "$(date) distillation done: $(grep '"loss"' $O/log.jsonl | tail -1 | cut -c1-120)"
scripts/train.sh $O-len "${DATA[@]}" "${MODEL[@]}" --init-from $O/last.pt --train-only len_q length \
	--batch 32 --max-tokens 32000 --lr 1e-3 --warmup 200 --steps 6000 --decay-frac 0.3 --eval-every 1000 --val-size 256 \
	--len-frac 1.0 --len-ramp 1 --len-weight 1.0 --cond-drop 0 --text-drop 0 --prompt-drop 0 --prompt-s 3 10 --long-p 0
name=$(basename $O)
scripts/evaluate.sh $O-len/last.pt $E/$name $E/long-soup0.7-regular --cfg 0 --cfg-prompt 0 | grep "vs\|  all\|  en "
[ -f $E/$name-long/scores.json ] || MIOPEN_FIND_MODE=FAST $P -m vall_e.evalset run --set evalsets/v3-long.json --ckpt $O-len/last.pt --out $E/$name-long \
	--length predict --cfg 0 --cfg-prompt 0 --gt $E/long-gt > $E/$name-long.log 2>&1
$P -m vall_e.evalset compare $E/long-soup0.7 $E/$name-long --gt $E/long-gt 2>&1 | grep "==\|  all\|  en "
echo "$(date) distill chain done"
