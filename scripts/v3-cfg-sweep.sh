#!/bin/bash
# Guidance strength beyond 3.5 (SIM rose 0.592 -> 0.656 -> 0.677 for 1.5 -> 2.5 -> 3.5 with no WER cost at 3.5).
cd "$(dirname "$0")/.."
E=training/v3-evals
P="venv/bin/python -m vall_e.v3.evalset"
for c in 4.5 6.0; do
	[ -f $E/sim-cfg$c/scores.json ] || MIOPEN_FIND_MODE=FAST $P run --set evalsets/v3-eval.json --out $E/sim-cfg$c --ckpt training/v3-main5p-len/last.pt --length predict --cfg $c > $E/sim-cfg$c.log 2>&1
done
$P report $E/sim-cfg3.5 $E/sim-cfg4.5 $E/sim-cfg6.0 --gt $E/gt 2>&1 | grep "==\|  all\|  en "
for c in 3.5 4.5 6.0; do $P compare $E/flagship-pred1.0 $E/sim-cfg$c --gt $E/gt 2>&1 | grep "==\|  all"; done
echo "$(date) cfg sweep done"
