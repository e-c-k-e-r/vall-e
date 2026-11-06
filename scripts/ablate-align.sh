#!/bin/bash
# Small from-scratch ablation of alignment aids (41M, 20k steps at batch 32, the ladder's filtered-store rung): the
# baseline, REPA (a shallow block aligned to mHuBERT-147 features of the target audio) and a 4-block ConvNeXt text
# pre-net. Each is scored on the eval set at its reference lengths (no length head) and against the baseline; the
# logs carry val/text_use. All three train on the same fixed 60k-utterance subset, the one whose REPA targets
# vall_e/repa.py baked (decoding audio and running HuBERT inside training was ~20x slower on ROCm). Resumable per arm.
cd "$(dirname "$0")/.."
. scripts/local.sh
PY=${VALL_E_EVAL_PY:-venv/bin/python}
E=training/evals; R=${REPA_FEATS:-training/repa-mhubert}
[ -f $R/done ] || $PY -m vall_e.repa bake --store $S "${GROUPS_ARGS[@]}" --out $R --count ${REPA_COUNT:-60000} >> $R.log 2>&1
BASE=(--store $S "${GROUPS_ARGS[@]}" --min-rate 13 --min-rate-groups LibriVox Audiobooks --head flow --steps 20000 --batch 32
	--workers 6 --eval-every 2000 --val-size 128 --mask-ratio cos --cond-drop 0.1 --text-drop 0 --prompt-drop 0 --max-tokens 32000 --compile)
arm() {  # arm <name> <train.py args...>
	local name=$1; shift
	local out=training/v3-align-$name
	if ! grep -q '"step": 20000' $out/log.jsonl 2>/dev/null; then
		echo "$(date) $name: training"
		scripts/train.sh $out "${BASE[@]}" "$@" > /dev/null 2>&1
	fi
	echo "$(date) $name: $(grep text_use $out/log.jsonl | tail -1 | grep -o '"val/text_use": [-0-9.e]*')"
	[ -f $E/align-$name/scores.json ] || MIOPEN_FIND_MODE=FAST $PY -m vall_e.evalset run --set evalsets/v3-eval.json --ckpt $out/last.pt \
		--out $E/align-$name --length ref --cfg 6.0 --gt $E/gt > $E/align-$name.log 2>&1
	$PY -m vall_e.evalset report $E/align-$name --gt $E/gt 2>&1 | grep "  all\|  en " | cut -c1-200
	[ $name != base ] && $PY -m vall_e.evalset compare $E/align-base $E/align-$name --gt $E/gt 2>&1 | grep "  all\|  en "
}
arm base --items-from $R/ids.npy
arm repa --repa-weight 0.5 --repa-feats $R
arm prenet --text-prenet 4 --items-from $R/ids.npy
echo "$(date) alignment ablation done"
