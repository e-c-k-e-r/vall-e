#!/bin/bash
# Cumulative additions on R2f's recipe (single additions each kept text_use at 3.9-4.7%; all of them together fell to
# 1.2%): adds them in order of need until text_use collapses. 41M / 20k steps / batch 32, same store as the ladder.
# usage: nohup scripts/v3-ablate-cumulative.sh >> training/v3-after-main5.log 2>&1 &
cd "$(dirname "$0")/.."
S=data/store
ALL=(Audiobooks LibriTTS-R LibriVox Emilia "GROUP" GROUP GROUP GROUP GROUP GROUP GROUP GROUP GROUP "GROUP" "GROUP"
	"GROUP" GROUP GROUP GROUP "GROUP" GROUP GROUP)
BASE=(--store $S --groups "${ALL[@]}" --min-rate 13 --min-rate-groups LibriVox Audiobooks
	--head flow --steps 20000 --batch 32 --compile --workers 6 --eval-every 2000 --val-size 128
	--mask-ratio cos --cond-drop 0.1 --text-drop 0 --prompt-drop 0 --max-tokens 32000)
rung() {  # rung <name> <extra train.py args...>
	local name=$1; shift
	local d=training/v3-abl2-$name
	[ -f $d/diag.log ] && return
	echo "$(date) $name: training"
	scripts/v3-proto-run.sh $d "${BASE[@]}" "$@"
	venv/bin/python scratch/v3_diag.py $d/last.pt $d/diag 16 2>&1 | grep "^#\|Error" > $d/diag.log.tmp
	mv $d/diag.log.tmp $d/diag.log
	echo "$(date) $name: text_use $(python3 -c "import json; print(round([json.loads(l) for l in open('$d/log.jsonl') if 'val/text_use' in l][-1]['val/text_use'], 4))") | $(grep -h "nar_cfg2.5 \|flowlarge" $d/diag.log | tr -s ' ' | tr '\n' ' ')"
}
rung c1-len-accent --len-frac 0.05 --accent
rung c2-prompt10 --len-frac 0.05 --accent --prompt-s 3 10
rung c3-similar --len-frac 0.05 --accent --prompt-s 3 10 --similar-p 0.75
echo "$(date) cumulative ablations done"
