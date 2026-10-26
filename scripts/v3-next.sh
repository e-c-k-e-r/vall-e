#!/bin/bash
# After the 3-10 s prompt fine-tune: (1) whether the old NeMo v2 model reads the text (curriculum vs architecture),
# (2) the CTC alignment loss at ablation scale: R2f + CTC (does it speed up text reading?) and C3 + CTC (does it let
# the additions coexist? C3 alone fell to 1.9% text_use).
# usage: nohup scripts/v3-next.sh >> training/v3-next.log 2>&1 &
cd "$(dirname "$0")/.."
while ps -eo args | grep -v grep | grep -q "v3-main5-prompt10.sh\|v3-main5-len.sh"; do sleep 120; done
while ps -eo args | grep -v grep | grep -q "vall_e.v3.train\|vall_e.v3.sample\|v3_diag.py"; do sleep 60; done
if [ ! -f training/llama-nemo/v3val-check/scores.json ]; then
	echo "$(date) v2-nemo check"
	timeout 5400 venv/bin/python scratch/vallef/v2nemo_on_v3val.py > training/llama-nemo/v3val-check.log 2>&1
	echo "$(date) v2-nemo: $(grep -h '^v2-nemo\|Error' training/llama-nemo/v3val-check.log | tail -2 | tr '\n' ' ')"
fi
S=data/store
ALL=(Audiobooks LibriTTS-R LibriVox Emilia "GROUP" GROUP GROUP GROUP GROUP GROUP GROUP GROUP GROUP "GROUP" "GROUP"
	"GROUP" GROUP GROUP GROUP "GROUP" GROUP GROUP)
BASE=(--store $S --groups "${ALL[@]}" --min-rate 13 --min-rate-groups LibriVox Audiobooks
	--head flow --steps 20000 --batch 32 --compile --workers 6 --eval-every 2000 --val-size 128
	--mask-ratio cos --cond-drop 0.1 --text-drop 0 --prompt-drop 0 --max-tokens 32000)
rung() {  # rung <name> <extra train.py args...>
	local name=$1; shift
	local d=training/v3-abl3-$name
	[ -f $d/diag.log ] && return
	echo "$(date) $name: training"
	scripts/v3-proto-run.sh $d "${BASE[@]}" "$@"
	venv/bin/python scratch/v3_diag.py $d/last.pt $d/diag 16 2>&1 | grep "^#\|Error" > $d/diag.log.tmp
	mv $d/diag.log.tmp $d/diag.log
	echo "$(date) $name: text_use $(python3 -c "import json; print(round([json.loads(l) for l in open('$d/log.jsonl') if 'val/text_use' in l][-1]['val/text_use'], 4))") | $(grep -h "nar_cfg2.5 \|flowlarge" $d/diag.log | tr -s ' ' | tr '\n' ' ')"
}
rung r2f-ctc --ctc-weight 0.1
rung c3-ctc --ctc-weight 0.1 --len-frac 0.05 --accent --prompt-s 3 10 --similar-p 0.75
echo "$(date) CTC ablations done"
