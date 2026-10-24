#!/bin/bash
# Which change since v3-flow stops the model reading the text? Small model (41M, v3-flow's size), 20k steps per rung,
# each rung adding one group of changes; every eval logs val/text_use (loss increase with another item's text), and
# each rung ends with a 16-item sampled WER check. Resumable: finished rungs (diag log present) are skipped.
# usage: nohup scripts/v3-ablate.sh >> training/v3-ablate.log 2>&1 &
cd "$(dirname "$0")/.."
S=data/store
ALL=(Audiobooks LibriTTS-R LibriVox Emilia "GROUP" GROUP GROUP GROUP GROUP GROUP GROUP GROUP GROUP "GROUP" "GROUP"
	"GROUP" GROUP GROUP GROUP "GROUP" GROUP GROUP)
BASE=(--head flow --steps 20000 --batch 32 --compile --workers 6 --eval-every 2000 --val-size 128
	--mask-ratio cos --cond-drop 0.1 --text-drop 0 --prompt-drop 0 --max-tokens 32000)

rung() {  # rung <name> <train.py args...>
	local name=$1; shift
	local out=training/v3-abl-$name
	[ -f $out/diag.log ] && return
	echo "$(date) $name: training"
	scripts/v3-proto-run.sh $out "${BASE[@]}" "$@"
	echo "$(date) $name: sampled check"
	venv/bin/python scratch/v3_diag.py $out/last.pt $out/diag 16 2>&1 | grep "^#\|Error" > $out/diag.log.tmp
	mv $out/diag.log.tmp $out/diag.log
	echo "$(date) $name: $(grep -h text_use $out/log.jsonl | tail -1 | grep -o '"val/text_use": [-0-9.e]*') | $(grep -v flowlarge $out/diag.log | tr '\n' ' ')"
}

# R0: v3-flow's recipe on its JSON index (LibriTTS-R + 20k Emilia/EN speakers): the baseline that reads the text
mkdir -p training/v3-abl-r0-json; [ -f training/v3-abl-r0-json/index.json ] || cp training/v3-flow-large/index.json training/v3-abl-r0-json/
rung r0-json
# R1: the same groups through the store
rung r1-store --store $S --groups LibriTTS-R Emilia/EN
# R2: the whole store, natural mix
rung r2-full --store $S --groups "${ALL[@]}"
# R3: + 3-10 s prompts from the similar lists
rung r3-prompts --store $S --groups "${ALL[@]}" --prompt-s 3 10 --similar-p 0.75
# R4: + accent, LEN mode, noised prompts, within-group speaker balance, independent text / prompt drops (= v3-main4)
rung r4-main4 --store $S --groups "${ALL[@]}" --prompt-s 3 10 --similar-p 0.75 \
	--accent --len-frac 0.05 --noisy-p 0.1 --speaker-alpha 0.5 --text-drop 0.05 --prompt-drop 0.1
echo "$(date) ablation done"
