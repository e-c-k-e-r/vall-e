#!/bin/bash
# After the ablation ladder: starts the full-size flagship run (v3-main5) on the rate-filtered store with the richest
# recipe that passed (R4f, else R2f; pass = val/text_use >= 3% and sampled NAR WER within 2x v3-flow-large's on the
# same items), and stops it if val/text_use is still under 2% at step 10k. Starts nothing if neither rung passed.
# usage: nohup scripts/v3-main5-after-ablate.sh >> training/v3-main5-chain.log 2>&1 &
cd "$(dirname "$0")/.."
until grep -q "ablation done" training/v3-ablate.log 2>/dev/null; do sleep 120; done

passes() {  # passes <rung dir>
	local d=training/v3-abl-$1
	python3 - "$d" <<'EOF'
import json, re, sys
d = sys.argv[1]
use = [json.loads(l)["val/text_use"] for l in open(f"{d}/log.jsonl") if "val/text_use" in l]
diag = open(f"{d}/diag.log").read()
wer = lambda k: float(re.search(rf"# {k}\s+WER ([0-9.]+)", diag).group(1))
try:
	ok = use[-1] >= 0.03 and wer("nar_cfg2.5") <= 2 * wer("flowlarge_cfg2.5")
	print(f"{d}: text_use {use[-1]:.3f}, WER {wer('nar_cfg2.5'):.3f} vs v3-flow-large {wer('flowlarge_cfg2.5'):.3f} -> {'pass' if ok else 'fail'}", file=sys.stderr)
except Exception as e:
	print(f"{d}: unreadable results ({e})", file=sys.stderr); ok = False
sys.exit(0 if ok else 1)
EOF
}

S=data/store
ALL=(Audiobooks LibriTTS-R LibriVox Emilia "GROUP" GROUP GROUP GROUP GROUP GROUP GROUP GROUP GROUP "GROUP" "GROUP"
	"GROUP" GROUP GROUP GROUP "GROUP" GROUP GROUP)
if passes r4f-main4; then
	EXTRA=(--prompt-s 3 10 --similar-p 0.75 --accent --len-frac 0.05 --noisy-p 0.1 --speaker-alpha 0.5 --text-drop 0.05 --prompt-drop 0.1)
	echo "$(date) R4f passed: v3-main5 gets v3-main4's additions"
elif passes r2f-filtered; then
	EXTRA=(--text-drop 0 --prompt-drop 0)
	echo "$(date) only R2f passed: v3-main5 gets the plain recipe"
else
	echo "$(date) neither filtered rung passed: not starting v3-main5"; exit 1
fi

out=training/v3-main5
# the text-use gate: stop the run if it isn't reading the text by step 10k
(
	until grep -q '"step": 10000}' $out/log.jsonl 2>/dev/null; do sleep 120; done
	use=$(python3 -c "import json; print([r for r in map(json.loads, open('$out/log.jsonl')) if r.get('step') == 10000 and 'val/text_use' in r][-1]['val/text_use'])")
	if python3 -c "import sys; sys.exit(0 if float('$use') < 0.02 else 1)"; then
		echo "$(date) v3-main5 text_use $use < 0.02 at step 10k: stopping it"
		for pat in "v3-proto-run.sh $out" "vall_e.v3.train --out $out"; do
			ps -eo pid,args | grep -F "$pat" | grep -v grep | awk '{print $1}' | xargs -r kill; sleep 5
		done
		sleep 20; ps -eo pid,comm | awk '$2=="pt_data_worker"{print $1}' | xargs -r kill
	else
		echo "$(date) v3-main5 text_use $use at step 10k: reading the text, continuing"
	fi
) &

echo "$(date) training $out"
scripts/v3-proto-run.sh $out --store $S --groups "${ALL[@]}" --min-rate 13 --min-rate-groups LibriVox Audiobooks \
	--head flow --d-model 1024 --layers 16 --heads 16 --batch 16 --max-tokens 16000 \
	--lr 3e-4 --warmup 1000 --steps 300000 --decay-frac 0.2 --compile --workers 6 --keep-every 50000 \
	--mask-ratio cos --cond-drop 0.1 "${EXTRA[@]}"
echo "$(date) training finished"
