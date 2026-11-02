#!/bin/bash
# A decay probe for v3-main6: after its step-100k eval, pause it, branch a 10k-step linear decay to zero from
# step-100000.pt (WSD's stable phase scores below an annealed model, so this estimates what the run is worth), score
# that against the flagship, and resume the main run.
# usage: nohup scripts/v3-main6-probe.sh >> training/v3-main6-evals.log 2>&1 &
cd "$(dirname "$0")/.."
E=training/v3-evals
P="venv/bin/python -m vall_e.v3.evalset"
out=training/v3-main6
probe=training/v3-main6-probe100k
until grep -q "main6-100000-sp2 vs" training/v3-main6-evals.log 2>/dev/null; do sleep 120; done
sleep 120  # let the watcher resume the main run first, then pause it again
for pat in "v3-main6-chain.sh" "v3-proto-run.sh $out" "vall_e.v3.train --out $out"; do
	ps -eo pid,args | grep -F "$pat" | grep -v grep | awk '{print $1}' | xargs -r kill; sleep 5
done
sleep 20; ps -eo pid,comm | awk '$2=="pt_data_worker"{print $1}' | xargs -r kill; sleep 10
echo "$(date) decay probe from step 100k"
S=data/store
ALL=(Audiobooks LibriTTS-R LibriVox Emilia "GROUP" GROUP GROUP GROUP GROUP GROUP GROUP GROUP GROUP "GROUP" "GROUP"
	"GROUP" GROUP GROUP GROUP "GROUP" GROUP GROUP)
scripts/v3-proto-run.sh $probe --store $S --groups "${ALL[@]}" --min-rate 13 --min-rate-groups LibriVox Audiobooks \
	--init-from $out/step-100000.pt --head flow --d-model 1024 --layers 16 --heads 16 --batch 16 --max-tokens 16000 \
	--lr 1e-4 --warmup 200 --steps 10000 --decay-frac 0.98 --compile --workers 6 --eval-every 2000 \
	--mask-ratio cos --cond-drop 0.1 --prompt-drop 0.1 --text-drop 0 --prompt-s 3 10
for n in main6-probe100k main6-probe100k-sp2; do
	extra=""; [ $n = main6-probe100k-sp2 ] && extra="--cfg-prompt 2.0"
	[ -f $E/$n/scores.json ] || MIOPEN_FIND_MODE=FAST $P run --set evalsets/v3-eval.json --out $E/$n --ckpt $probe/last.pt --length ref --cfg 6.0 $extra > $E/$n.log 2>&1
	$P compare $E/flagship-ref6 $E/$n --gt $E/gt 2>&1 | grep "==\|  all\|  en "
done
nohup scripts/v3-main6-chain.sh >> training/v3-main6-chain.log 2>&1 &
echo "$(date) decay probe done, main run resumed"
