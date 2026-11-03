#!/bin/bash
# After v3-main6 finishes: a length head on the frozen final model, then the final paired evaluation with predicted
# lengths (plain, speaker guidance 1 / 2 / 3), against the flagship and the step-100k decay probe.
# usage: nohup scripts/v3-main6-final.sh >> training/v3-main6-final.log 2>&1 &
cd "$(dirname "$0")/.."
E=training/v3-evals
P="venv/bin/python -m vall_e.v3.evalset"
until grep -q "main6 evals done" training/v3-main6-evals.log 2>/dev/null; do sleep 120; done
while ps -eo args | grep -v grep | grep -q "vall_e.v3.train\|vall_e.v3.evalset run"; do sleep 60; done
S=data/store
ALL=(Audiobooks LibriTTS-R LibriVox Emilia "GROUP" GROUP GROUP GROUP GROUP GROUP GROUP GROUP GROUP "GROUP" "GROUP"
	"GROUP" GROUP GROUP GROUP "GROUP" GROUP GROUP)
echo "$(date) length head on v3-main6"
scripts/v3-proto-run.sh training/v3-main6-len --store $S --groups "${ALL[@]}" --min-rate 13 --min-rate-groups LibriVox Audiobooks \
	--init-from training/v3-main6/last.pt --train-only len_q length --head flow --d-model 1024 --layers 16 --heads 16 --compile --workers 6 \
	--batch 32 --max-tokens 32000 --lr 1e-3 --warmup 200 --steps 6000 --decay-frac 0.3 --eval-every 1000 --val-size 256 \
	--len-frac 1.0 --len-ramp 1 --len-weight 1.0 --cond-drop 0 --text-drop 0 --prompt-drop 0 --prompt-s 3 10
echo "$(date) length head: $(grep '\[eval\]' training/v3-main6-len/run.log | tail -1 | grep -o 'val@len/rel_err [0-9.]*\|val/text_use [0-9.]*' | tr '\n' ' ')"
M=training/v3-main6-len/last.pt
run() { local name=$1; shift; [ -f $E/$name/scores.json ] || MIOPEN_FIND_MODE=FAST $P run --set evalsets/v3-eval.json --out $E/$name "$@" > $E/$name.log 2>&1; }
run final-pred --ckpt $M --length predict
for sp in 1.0 2.0 3.0; do run final-pred-sp$sp --ckpt $M --length predict --cfg-prompt $sp; done
run probe100k-ref-sp2 --ckpt training/v3-main6-probe100k/last.pt --length ref --cfg-prompt 2.0
$P report $E/gt $E/flagship-pred1.0 $E/sim-cfg6.0 $E/final-pred $E/final-pred-sp1.0 $E/final-pred-sp2.0 $E/final-pred-sp3.0 --gt $E/gt 2>&1 | grep "==\|  [a-z]"
for b in final-pred final-pred-sp1.0 final-pred-sp2.0 final-pred-sp3.0; do $P compare $E/sim-cfg6.0 $E/$b --gt $E/gt 2>&1 | grep "==\|  all\|  en "; done
$P compare $E/main6-probe100k-sp2 $E/main6-300000-sp2 --gt $E/gt 2>&1 | grep "==\|  all\|  en "
echo "$(date) final done"
