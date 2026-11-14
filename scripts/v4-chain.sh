#!/bin/bash
# v4: the recipe (scripts/recipe.sh) from scratch at 746M (d1536 x 24, activation checkpointing), on the main store
# (rephonemized with the current front-end) plus the Genshin store, games and Genshin at 15% of draws from step 0,
# the held-out game speakers and Genshin characters (data/holdout-speakers-v2.json) never trained on. Long-form
# targets (up to 30 s, at their natural rate) are folded into the second stage instead of a separate fine-tune. About 8.3 days.
#   GENSHIN=<store> scripts/v4-chain.sh        SMOKE=1: the base stage's first 520 steps into training/v4-smoke
# Stages resume (scripts/train.sh) and are skipped once their last.pt is in place.
cd "$(dirname "$0")/.."
. scripts/local.sh
G=${GENSHIN:?the Genshin store}
E=training/evals; P=${VALL_E_EVAL_PY:-venv/bin/python}
export MIOPEN_FIND_MODE=FAST
OVER=(TF2 HL2 HL Portal P3 P4 KH PPP housamo "Deus Ex" "System Shock" SA2 FFVIIR Cyberpunk "Elden Ring" RE4Make "Sam and Max" "Stanley Parable" Genshin)
MODEL=(--head flow --d-model 1536 --layers 24 --heads 24 --compile --grad-ckpt --workers 6)
DATA=(--store $S "${GROUPS_ARGS[@]}" --extra-store $G --min-rate 13 --min-rate-groups LibriVox Audiobooks
	--holdout-speakers data/holdout-speakers-v2.json --oversample-groups "${OVER[@]}" --oversample-p 0.15)
NAR=(--batch 16 --max-tokens 16000 --mask-ratio cos --cond-drop 0.1 --text-drop 0)
BASE=(--prompt-drop 0 --lr 2e-4 --warmup 2000 --decay-frac 0.2)
# long-form at its natural rate (7.7% of utterances over 12 s, ~21% of the audio): already ~2x the long training of
# v1.2's fine-tune over 300k steps; --long-p 0.25 would spend half the stage's audio on them at the cost of coverage
MORE=(--prompt-drop 0.1 --prompt-s 3 10 --max-s 30)
if [ -n "$SMOKE" ]; then  # one log interval of the base stage, then of the long-form stage from it (the most memory)
	scripts/train.sh training/v4-smoke "${DATA[@]}" "${MODEL[@]}" "${NAR[@]}" "${BASE[@]}" --steps 520 --eval-every 1000000
	scripts/train.sh training/v4-smoke-more "${DATA[@]}" "${MODEL[@]}" "${NAR[@]}" "${MORE[@]}" --init-from training/v4-smoke/last.pt \
		--lr 7e-5 --warmup 1000 --steps 520 --eval-every 1000000
	exit
fi
done_at() {  # <run dir> <steps>: the run's log reached its last step
	[ -f $1/last.pt ] && [ "$(python3 -c "import json; print(int(json.loads(open('$1/log.jsonl').readlines()[-1])['step']))" 2>/dev/null)" = $2 ]
}
run() {  # <set> <ckpt> <out> [gt]: retried (ROCm segfaults; a run resumes)
	for try in 1 2 3 4; do
		[ -f $3/scores.json ] && return
		$P -m vall_e.evalset run --set $1 --ckpt $2 --out $3 --length predict --cfg 6.0 --cfg-prompt 2.0 ${4:+--gt $4} >> $3.log 2>&1
	done
}
evals() {  # <ckpt> <name>: the main, long, held-out game and held-out Genshin sets (the last two with 3-clip prompts too)
	run evalsets/v3-eval.json $1 $E/main-$2 $E/gt
	run evalsets/v3-long.json $1 $E/long-$2 $E/long-gt
	run evalsets/v3-games-holdout.json $1 $E/hgames-$2 $E/holdout-gt
	run training/prompts-games-concat3.json $1 $E/hgames-c3-$2 $E/holdout-gt
	run evalsets/v3-genshin-holdout.json $1 $E/genshin-$2 $E/genshin-gt
	run training/prompts-genshin-concat3.json $1 $E/genshin-c3-$2 $E/genshin-gt
}
# baseline: v1.3 on the rephonemized store (the earlier runs read the old phonemes)
evals data/models/nemo-44khz-nar-flow-225m-v1.3.sft v1.3r
echo "$(date) baselines done"
done_at training/v4-base 300000 || {
	# the recipe's gate: the 225M run was at 0.023 by 30k (0.005 at 20k, so a 20k gate would stop a healthy run)
	nohup scripts/textuse-gate.sh training/v4-base 30000 0.015 >> training/v4-base-gate.log 2>&1 &
	scripts/train.sh training/v4-base "${DATA[@]}" "${MODEL[@]}" "${NAR[@]}" "${BASE[@]}" --steps 300000 --keep-every 100000 > /dev/null 2>&1; }
done_at training/v4-base 300000 || { echo "$(date) base FAILED"; exit 1; }
echo "$(date) base done"
done_at training/v4-prompts 30000 || scripts/train.sh training/v4-prompts "${DATA[@]}" "${MODEL[@]}" "${NAR[@]}" --prompt-drop 0 --prompt-s 3 10 \
	--init-from training/v4-base/last.pt --lr 7e-5 --warmup 500 --steps 30000 --decay-frac 0.3 > /dev/null 2>&1
echo "$(date) prompts done"
done_at training/v4-more 300000 || {
	nohup scripts/textuse-gate.sh training/v4-more 10000 0.04 >> training/v4-more-gate.log 2>&1 &
	scripts/train.sh training/v4-more "${DATA[@]}" "${MODEL[@]}" "${NAR[@]}" "${MORE[@]}" \
		--init-from training/v4-prompts/last.pt --lr 7e-5 --warmup 1000 --steps 300000 --decay-frac 0.2 --keep-every 100000 > /dev/null 2>&1; }
echo "$(date) more done"
done_at training/v4-len 6000 || scripts/train.sh training/v4-len "${DATA[@]}" "${MODEL[@]}" --init-from training/v4-more/last.pt --train-only len_q length \
	--batch 32 --max-tokens 32000 --lr 1e-3 --warmup 200 --steps 6000 --decay-frac 0.3 --eval-every 1000 --val-size 256 \
	--len-frac 1.0 --len-ramp 1 --len-weight 1.0 --cond-drop 0 --text-drop 0 --prompt-drop 0 --prompt-s 3 10 --max-s 30 > /dev/null 2>&1
echo "$(date) length head done"
evals training/v4-len/last.pt v4
for s in "main gt" "long long-gt" "hgames holdout-gt" "hgames-c3 holdout-gt" "genshin genshin-gt" "genshin-c3 genshin-gt"; do
	set -- $s
	echo "== v4, $1 (A: v1.3 on the same phonemes):"; $P -m vall_e.evalset compare $E/$1-v1.3r $E/$1-v4 --gt $E/$2 2>&1 | grep "  all\|  en \|  ja "
done
echo "$(date) v4 chain done"
