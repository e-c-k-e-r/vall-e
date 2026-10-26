#!/bin/bash
# After the length head: v3-main5 fine-tuned on 3-10 s prompts (longer prompts raised SIM in the prompt-length test),
# stopped if val/text_use falls under 4% at 10k (v3-main5: ~5.5%), then scored against v3-main5 with 10 s prompts.
# usage: nohup scripts/v3-main5-prompt10.sh >> training/v3-main5-prompt10.log 2>&1 &
cd "$(dirname "$0")/.."
until grep -q "length head done" training/v3-main5-len.log 2>/dev/null; do sleep 60; done
S=data/store
ALL=(Audiobooks LibriTTS-R LibriVox Emilia "GROUP" GROUP GROUP GROUP GROUP GROUP GROUP GROUP GROUP "GROUP" "GROUP"
	"GROUP" GROUP GROUP GROUP "GROUP" GROUP GROUP)
out=training/v3-main5-prompt10
nohup scripts/v3-textuse-gate.sh $out 10000 0.04 >> training/v3-main5-prompt10.log 2>&1 &
scripts/v3-proto-run.sh $out --store $S --groups "${ALL[@]}" --min-rate 13 --min-rate-groups LibriVox Audiobooks \
	--init-from training/v3-main5/last.pt \
	--head flow --d-model 1024 --layers 16 --heads 16 --batch 16 --max-tokens 16000 \
	--lr 1e-4 --warmup 500 --steps 30000 --decay-frac 0.3 --compile --workers 6 --keep-every 50000 \
	--mask-ratio cos --cond-drop 0.1 --text-drop 0 --prompt-drop 0 --prompt-s 3 10
step=$(grep -o '"step": [0-9]*}' $out/log.jsonl | tail -1 | grep -o '[0-9]*')
echo "$(date) fine-tune ended at step $step"
[ "$step" = "30000" ] || exit 0  # stopped by the gate
for m in v3-main5:training/v3-main5/last.pt prompt10:$out/last.pt; do
	name=${m%%:*}; ckpt=${m#*:}; d=$out/eval-$name
	[ -f $d/scores.json ] || venv/bin/python -m vall_e.v3.sample --ckpt $ckpt --out $d --per-language 6 --prompt-seconds 10 --score > $d.log 2>&1
	echo "$(date) $name, 10 s prompts: $(grep -h '^gen' $d.log)"
done
venv/bin/python scratch/v3_diag.py $out/last.pt $out/en32 32 2>&1 | grep "^#" | grep -v "refs" > $out/en32.log
echo "$(date) prompt10 en32: $(tr '\n' ' ' < $out/en32.log)"
