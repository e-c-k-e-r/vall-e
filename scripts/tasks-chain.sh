#!/bin/bash
# The tasks fine-tune and its checks: after the noisy-target bake, (1) the noisy baseline, (2) a fine-tune of the
# flagship with span masking (editing / continuation), noise suppression and speech removal, stopped if it stops
# reading the text, (3) a fresh length head on it, (4) a TTS regression check against the flagship, noise suppression
# against the noisy baseline, and the edit / continue test again.
# usage: nohup scripts/tasks-chain.sh >> training/tasks-chain.log 2>&1 &
cd "$(dirname "$0")/.."
S=${VALL_E_STORE:-data/store}
E=training/evals
P="venv/bin/python -m vall_e.evalset"
GROUPS_=(Audiobooks LibriTTS-R LibriVox Emilia "GROUP" GROUP GROUP GROUP GROUP GROUP GROUP GROUP GROUP "GROUP" "GROUP"
	"GROUP" GROUP GROUP GROUP "GROUP" GROUP GROUP)
DATA=(--store $S --groups "${GROUPS_[@]}" --min-rate 13 --min-rate-groups LibriVox Audiobooks)
MODEL=(--head flow --d-model 1024 --layers 16 --heads 16 --compile --workers 6)
until [ -f $S/noisy-targets/source.npy ]; do sleep 120; done
while ps -eo args | grep -v grep | grep -q "vall_e.noisy"; do sleep 30; done
echo "$(date) noisy-targets ready: $(python3 -c "import numpy as np; print(len(np.load('$S/noisy-targets/source.npy')))") pairs"

[ -f $E/noisy/scores.json ] || $P run --set evalsets/v3-eval.json --ckpt none --task noisy --out $E/noisy --gt $E/gt > $E/noisy.log 2>&1
$P report $E/noisy --gt $E/gt 2>&1 | grep "==\|  all\|  en "

out=training/v3-tasks
nohup scripts/textuse-gate.sh $out 10000 0.04 >> training/tasks-chain.log 2>&1 &
echo "$(date) fine-tune"
scripts/train.sh $out "${DATA[@]}" "${MODEL[@]}" --init-from training/v3-main6-len/last.pt \
	--batch 16 --max-tokens 16000 --mask-ratio cos --cond-drop 0.1 --text-drop 0 --prompt-drop 0.1 --prompt-s 3 10 \
	--span-p 0.3 --ns-p 0.1 --sr-p 0.05 --lr 1e-4 --warmup 500 --steps 40000 --decay-frac 0.3
step=$(grep -o '"step": [0-9]*}' $out/log.jsonl | tail -1 | grep -o '[0-9]*')
echo "$(date) fine-tune ended at step $step, text_use $(python3 -c "import json; print(round([json.loads(l) for l in open('$out/log.jsonl') if 'val/text_use' in l][-1]['val/text_use'], 4))")"
[ "$step" = "40000" ] || exit 0

scripts/train.sh $out-len "${DATA[@]}" "${MODEL[@]}" --init-from $out/last.pt --train-only len_q length \
	--batch 32 --max-tokens 32000 --lr 1e-3 --warmup 200 --steps 6000 --decay-frac 0.3 --eval-every 1000 --val-size 256 \
	--len-frac 1.0 --len-ramp 1 --len-weight 1.0 --cond-drop 0 --text-drop 0 --prompt-drop 0 --prompt-s 3 10 --n-modes 5
echo "$(date) length head: $(grep '\[eval\]' $out-len/run.log | tail -1 | grep -o 'val@len/rel_err [0-9.]*' )"
M=$out-len/last.pt
scripts/evaluate.sh $M $E/tasks training/evals/flagship
[ -f $E/tasks-ns/scores.json ] || $P run --set evalsets/v3-eval.json --ckpt $M --task ns --out $E/tasks-ns --gt $E/gt > $E/tasks-ns.log 2>&1
$P compare $E/noisy $E/tasks-ns --gt $E/gt 2>&1 | grep "==\|  [a-z]"
CKPT=$M PYTHONPATH=$PWD venv/bin/python scratch/v3_edit_test.py $E/edit-test-tasks 40 2>&1 | grep "^#"
echo "$(date) tasks chain done"
