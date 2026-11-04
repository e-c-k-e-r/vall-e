#!/bin/bash
# Runs vall_e/train.py, resuming from last.pt after a crash (up to 20 times). Waits until the GPU's VRAM is mostly free
# first. $VALL_E_TRAIN_PY picks the interpreter (on the ROCm box: torch 2.13, as torch 2.7 segfaults in MIOpen).
# usage: scripts/train.sh <out dir> [train.py args...]
cd "$(dirname "$0")/.."
PY=${VALL_E_TRAIN_PY:-python}
out=$1; shift
mkdir -p "$out"
# a private temp dir: something cleaning /tmp deleted the DataLoader's multiprocessing sockets twice (2026-10-22)
export TMPDIR="$PWD/$out/tmp"; mkdir -p "$TMPDIR"

until [ "$(rocm-smi --showmemuse 2>/dev/null | grep -oP 'VRAM%\): \K[0-9]+')" -lt 20 ] 2>/dev/null; do sleep 30; done
echo "$(date) VRAM free, starting" >> "$out/run.log"

for attempt in $(seq 1 20); do
	MIOPEN_FIND_MODE=FAST PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$PWD" \
		"$PY" -m vall_e.train --out "$out" "$@" >> "$out/run.log" 2>&1
	rc=$?  # before anything else runs: "$(date)" below would reset $?
	[ $rc -eq 0 ] && break
	echo "$(date) exited with $rc (139 = segfault), resuming (attempt $attempt)" >> "$out/run.log"
	sleep 30
done
echo "$(date) done" >> "$out/run.log"
