#!/bin/bash
# The released models (the -lenfix exports: v2.0, v2.0-fast, v1.3) on the five eval sets at their own guidance, for the documentation's
# numbers; then promptless generation (--prompt-seconds 0) on the main set with the old length heads against the new, the bug the new ones fix.
# Waits for scripts/len-promptless.sh.
#   scripts/release-evals.sh
cd "$(dirname "$0")/.."
E=training/evals; P=${VALL_E_EVAL_PY:-venv/bin/python}; M=data/models
export MIOPEN_FIND_MODE=FAST
while ps -eo args | grep -qE "^(/bin/)?bash scripts/len-promptless.sh"; do sleep 300; done
SETS=("main evalsets/v3-eval.json gt" "long evalsets/v3-long.json long-gt" "hgames evalsets/v3-games-holdout.json holdout-gt"
	"genshin evalsets/v3-genshin-holdout.json genshin-gt" "nikke evalsets/v3-nikke.json nikke-gt")
MODELS=("v2.0 nemo-44khz-nar-flow-746m-v2.0-lenfix 3.0 2.0" "v2.0-fast nemo-44khz-nar-flow-746m-v2.0-fast-lenfix 0 0"
	"v1.3 nemo-44khz-nar-flow-225m-v1.3-lenfix 6.0 2.0")
run() {  # out set gt ckpt cfg cfg_prompt [extra]
	local o=$1 s=$2 g=$3 c=$4 a=$5 b=$6; shift 6
	for t in 1 2 3; do [ -f $E/$o/scores.json ] && return; $P -m vall_e.evalset run --set $s --ckpt $c --out $E/$o --length predict --cfg $a --cfg-prompt $b --gt $E/$g "$@" >> $E/$o.log 2>&1; done
}
for m in "${MODELS[@]}"; do
	set -- $m; n=$1 x=$M/$2.sft c=$3 cp=$4
	[ -f $x ] || { echo "$(date) $x missing"; continue; }
	for s in "${SETS[@]}"; do set -- $s; run rel-$1-$n $2 $3 $x $c $cp; done
	echo "$(date) $n scored"
done
# promptless: the current exports' heads (A) against -lenfix (B)
for m in "${MODELS[@]}"; do
	set -- $m; n=$1 b=$2 c=$3 cp=$4 a=${2%-lenfix}
	run nop-main-$n-old evalsets/v3-eval.json gt $M/$a.sft $c $cp --prompt-seconds 0
	run nop-main-$n evalsets/v3-eval.json gt $M/$b.sft $c $cp --prompt-seconds 0
	echo "== promptless main: $n, current head (A) vs -lenfix (B)"; $P -m vall_e.evalset compare $E/nop-main-$n-old $E/nop-main-$n --gt $E/gt 2>&1 | grep "  all\|  en "
done
# the summaries (each run's log ends with its per-language lines)
for s in main long hgames genshin nikke; do
	for n in v2.0 v2.0-fast v1.3; do echo "== $s $n"; grep -E "^  (all|en|de|fr|ja|ko|zh) " $E/rel-$s-$n.log | tail -7; done
done
echo "$(date) release evals done"
