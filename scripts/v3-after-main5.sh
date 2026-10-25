#!/bin/bash
# After v3-main5: its final evaluation, then the single-addition ablations (R2f's recipe plus exactly one of v3-main4's
# additions per rung, 41M / 20k steps / batch 32 like the ladder, compared against R2f: text_use 4.6%, WER 26.5%).
# usage: nohup scripts/v3-after-main5.sh >> training/v3-after-main5.log 2>&1 &
cd "$(dirname "$0")/.."
out=training/v3-main5
until grep -q "done$" $out/run.log 2>/dev/null && ! ps -eo args | grep -v grep | grep -qF "vall_e.v3.train --out $out"; do sleep 120; done
echo "$(date) v3-main5 finished at $(grep -o '"step": [0-9]*}' $out/log.jsonl | tail -1)"

# final evaluation: 32 English items against v3-flow-large, then per language with SIM
ev=$out/final
mkdir -p $ev
[ -f $ev/en32.log ] || venv/bin/python scratch/v3_diag.py $out/last.pt $ev/en32 32 2>&1 | grep "^#\|Error" > $ev/en32.log
echo "$(date) en32: $(grep -h "nar_cfg2.5 \|flowlarge" $ev/en32.log | tr '\n' ' ')"
[ -f $ev/lang/scores.json ] || venv/bin/python -m vall_e.v3.sample --ckpt $out/last.pt --out $ev/lang --per-language 6 --score > $ev/lang.log 2>&1
echo "$(date) per-language: $(grep -h '^gen' $ev/lang.log)"

# single-addition ablations
S=data/store
ALL=(Audiobooks LibriTTS-R LibriVox Emilia "GROUP" GROUP GROUP GROUP GROUP GROUP GROUP GROUP GROUP "GROUP" "GROUP"
	"GROUP" GROUP GROUP GROUP "GROUP" GROUP GROUP)
BASE=(--store $S --groups "${ALL[@]}" --min-rate 13 --min-rate-groups LibriVox Audiobooks
	--head flow --steps 20000 --batch 32 --compile --workers 6 --eval-every 2000 --val-size 128
	--mask-ratio cos --cond-drop 0.1 --text-drop 0 --prompt-drop 0 --max-tokens 32000)
rung() {  # rung <name> <extra train.py args...>
	local name=$1; shift
	local d=training/v3-abl1-$name
	[ -f $d/diag.log ] && return
	echo "$(date) $name: training"
	scripts/v3-proto-run.sh $d "${BASE[@]}" "$@"
	venv/bin/python scratch/v3_diag.py $d/last.pt $d/diag 16 2>&1 | grep "^#\|Error" > $d/diag.log.tmp
	mv $d/diag.log.tmp $d/diag.log
	echo "$(date) $name: text_use $(python3 -c "import json; print(round([json.loads(l) for l in open('$d/log.jsonl') if 'val/text_use' in l][-1]['val/text_use'], 4))") | $(grep -h "nar_cfg2.5 \|flowlarge" $d/diag.log | tr -s ' ' | tr '\n' ' ')"
}
rung a1-prompt10 --prompt-s 3 10
rung a2-similar --similar-p 0.75
rung a3-noisy --noisy-p 0.1
rung a4-accent --accent
rung a5-len --len-frac 0.05
rung a6-speaker --speaker-alpha 0.5
rung a7-drops --text-drop 0.05 --prompt-drop 0.1
echo "$(date) single-addition ablations done"
