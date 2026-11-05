#!/bin/bash
# Long-form fine-tune: targets up to 30 s (40% of draws over 12 s) from the averaged v3-main6, a new length head on the
# frozen result, then the long set (evalsets/v3-long.json, 15-30 s, one pass) and the regular set against the start.
cd "$(dirname "$0")/.."
. scripts/local.sh
INIT=${INIT:-training/v3-main6-avg3-len/last.pt}
E=training/evals; P=${VALL_E_EVAL_PY:-venv/bin/python}
MODEL=(--head flow --d-model 1024 --layers 16 --heads 16 --compile --workers 6)
DATA=(--store $S "${GROUPS_ARGS[@]}" --min-rate 13 --min-rate-groups LibriVox Audiobooks)
long() {  # <ckpt> <out>: the long set at the defaults, one pass
	[ -f $E/long-gt/scores.json ] || MIOPEN_FIND_MODE=FAST $P -m vall_e.evalset run --set evalsets/v3-long.json --ckpt gt --out $E/long-gt > $E/long-gt.log 2>&1
	[ -f $2/scores.json ] || MIOPEN_FIND_MODE=FAST $P -m vall_e.evalset run --set evalsets/v3-long.json --ckpt $1 --out $2 --length ${3:-predict} \
		--cfg 6.0 --cfg-prompt 2.0 --gt $E/long-gt > $2.log 2>&1
	$P -m vall_e.evalset report $2 --gt $E/long-gt 2>&1 | grep "==\|  [a-z]"
}
echo "$(date) baselines"; long $INIT $E/long-base; long $INIT $E/long-base-ref ref
scripts/train.sh training/v3-long "${DATA[@]}" "${MODEL[@]}" --batch 16 --max-tokens 16000 --mask-ratio cos --cond-drop 0.1 --text-drop 0 \
	--prompt-drop 0.1 --prompt-s 3 10 --max-s 30 --long-p 0.4 --init-from $INIT --lr 5e-5 --warmup 500 --steps 30000 --decay-frac 0.3 \
	--keep-every 10000
echo "$(date) fine-tune done: $(tail -1 training/v3-long/log.jsonl | cut -c1-200)"
scripts/train.sh training/v3-long-len "${DATA[@]}" "${MODEL[@]}" --init-from training/v3-long/last.pt --train-only len_q length \
	--batch 32 --max-tokens 32000 --lr 1e-3 --warmup 200 --steps 6000 --decay-frac 0.3 --eval-every 1000 --val-size 256 \
	--len-frac 1.0 --len-ramp 1 --len-weight 1.0 --cond-drop 0 --text-drop 0 --prompt-drop 0 --prompt-s 3 10 --max-s 30
echo "$(date) length head done"
long training/v3-long-len/last.pt $E/long-ft; long training/v3-long-len/last.pt $E/long-ft-ref ref
$P -m vall_e.evalset compare $E/long-base $E/long-ft --gt $E/long-gt 2>&1 | grep "==\|  [a-z]"
$P -m vall_e.evalset compare $E/long-base-ref $E/long-ft-ref --gt $E/long-gt 2>&1 | grep "==\|  [a-z]"
scripts/evaluate.sh training/v3-long-len/last.pt $E/long-ft-regular $E/v3-grpo-len-base | grep "vs\|  all\|  en "
echo "$(date) long chain done"
