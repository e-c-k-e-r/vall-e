#!/bin/bash
# Continuation fine-tune (the old tts-c), tried on v1.3 (225M) before anything bigger: v2.0 never saw known audio
# followed by masked audio, and generating a chunk after the ones before it (TTS.stream) re-said the end of the known
# text ("mit ihr sprechen und mit ihr sprechen", WER +9 on the long set). --cont-p 0.3 makes 30% of items another
# utterance of the speaker known in front, its text in front with a space; the rest are the usual items. 20k steps.
# First v1.3's own streaming (does the 225M re-say too?), then the fine-tune, a length head, streaming against v1.3's,
# and the five eval sets against v1.3 on whole utterances (that it lost nothing).
#   scripts/v1.3-cont.sh          SMOKE=1: one log interval into training/v1.3-cont-smoke
cd "$(dirname "$0")/.."
. scripts/local.sh
T=training/v3-genshin-a0.5-len/last.pt; O=training/v1.3-cont; V=data/models/nemo-44khz-nar-flow-225m-v1.3.sft
E=training/evals; P=${VALL_E_EVAL_PY:-venv/bin/python}; G=~/Documents/datasets/training/dataset/44KHz-nemo-genshin
export MIOPEN_FIND_MODE=FAST
OVER=(TF2 HL2 HL Portal P3 P4 KH PPP housamo "Deus Ex" "System Shock" SA2 FFVIIR Cyberpunk "Elden Ring" RE4Make "Sam and Max" "Stanley Parable" Genshin)
MODEL=(--head flow --d-model 1024 --layers 16 --heads 16 --compile --workers 6)
DATA=(--store $S "${GROUPS_ARGS[@]}" --extra-store $G --min-rate 13 --min-rate-groups LibriVox Audiobooks --max-s 30
	--holdout-speakers data/holdout-speakers-v2.json --oversample-groups "${OVER[@]}" --oversample-p 0.15)
# continuation rows run up to 8 s past what the sampler counts, so fewer tokens per batch than the usual 16000
FT=(--init-from $T --cont-p 0.3 --cont-s 8 --batch 16 --max-tokens 12000 --mask-ratio cos --cond-drop 0.1 --text-drop 0
	--prompt-drop 0.1 --prompt-s 3 10 --lr 3e-5 --warmup 500 --decay-frac 0.3)
if [ -n "$SMOKE" ]; then
	scripts/train.sh training/v1.3-cont-smoke "${DATA[@]}" "${MODEL[@]}" "${FT[@]}" --steps 520 --eval-every 1000000
	exit
fi
stream() {  # mode ckpt out
	for t in 1 2 3; do [ -f $E/$3/scores.json ] && break; rm -rf $E/$3; PYTHONPATH=. $P training/datasets/stream_eval.py gen $1 $2 $E/$3 > $E/$3.log 2>&1; done
}
stream sent $V stream-v1.3-sent
stream sent-ind $V stream-v1.3-sent-ind
echo "$(date) v1.3 streaming done"
[ -f $O/last.pt ] && [ "$(grep -o '"step": [0-9]*' $O/log.jsonl | tail -1)" = '"step": 20000' ] || \
	scripts/train.sh $O "${DATA[@]}" "${MODEL[@]}" "${FT[@]}" --steps 20000 --keep-every 10000 > /dev/null 2>&1
[ "$(grep -o '"step": [0-9]*' $O/log.jsonl | tail -1)" = '"step": 20000' ] || { echo "$(date) cont FAILED"; exit 1; }
echo "$(date) continuation fine-tune done: $(grep '"loss"' $O/log.jsonl | tail -1 | cut -c1-120)"
[ -f $O-len/last.pt ] || scripts/train.sh $O-len "${DATA[@]}" "${MODEL[@]}" --init-from $O/last.pt --train-only len_q length \
	--batch 32 --max-tokens 32000 --lr 1e-3 --warmup 200 --steps 6000 --decay-frac 0.3 --eval-every 1000 --val-size 256 \
	--len-frac 1.0 --len-ramp 1 --len-weight 1.0 --cond-drop 0 --text-drop 0 --prompt-drop 0 --prompt-s 3 10 > /dev/null 2>&1
echo "$(date) length head done"
X=data/models/nemo-44khz-nar-flow-225m-v1.3-cont-candidate.sft
[ -f $X ] || $P -m vall_e.export $O-len/last.pt $X --cfg 6.0 --cfg-prompt 2.0
stream sent $X stream-v1.3c-sent
stream clause $X stream-v1.3c-clause
for b in "long-v1.3r stream-v1.3-sent" "long-v1.3r stream-v1.3-sent-ind" "stream-v1.3-sent stream-v1.3c-sent" "long-v1.3r stream-v1.3c-sent" "long-v1.3r stream-v1.3c-clause"; do
	set -- $b; echo "== long: $1 (A) vs $2 (B)"; $P -m vall_e.evalset compare $E/$1 $E/$2 --gt $E/long-gt 2>&1 | grep "  all\|  en "
done
for s in "main evalsets/v3-eval.json gt" "long evalsets/v3-long.json long-gt" "hgames evalsets/v3-games-holdout.json holdout-gt" "genshin evalsets/v3-genshin-holdout.json genshin-gt" "nikke evalsets/v3-nikke.json nikke-gt"; do
	set -- $s
	for t in 1 2 3 4; do [ -f $E/cont-$1-v1.3/scores.json ] && break; $P -m vall_e.evalset run --set $2 --ckpt $O-len/last.pt --out $E/cont-$1-v1.3 --length predict --cfg 6.0 --cfg-prompt 2.0 --gt $E/$3 >> $E/cont-$1-v1.3.log 2>&1; done
	echo "== $1: v1.3 (A) vs continuation fine-tune (B), whole utterances"; $P -m vall_e.evalset compare $E/$1-v1.3r $E/cont-$1-v1.3 --gt $E/$3 2>&1 | grep "  all\|  en \|  ja "
done
PYTHONPATH=. $P training/datasets/stream_eval.py bounds evalsets/v3-long.json $E/long-v1.3r $E/stream-v1.3-sent $E/stream-v1.3c-sent $E/stream-v1.3c-clause 2>&1 | grep -E "^(run|long-|stream)"
echo "$(date) v1.3 cont done"
