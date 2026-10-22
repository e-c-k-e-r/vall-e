#!/bin/bash
# Runs the v3 prototype (vall_e/v3/train.py) on the tcaptcha venv (torch 2.13; the vall-e venv's torch 2.7 segfaults
# in MIOpen here). Waits until the GPU's VRAM is mostly free, then trains, resuming from last.pt after a crash.
# usage: scripts/v3-proto-run.sh <out dir> [train.py args...]
cd "$(dirname "$0")/.."
PY=python
out=$1; shift
mkdir -p "$out"

until [ "$(rocm-smi --showmemuse 2>/dev/null | grep -oP 'VRAM%\): \K[0-9]+')" -lt 20 ] 2>/dev/null; do sleep 30; done
echo "$(date) VRAM free, starting" >> "$out/run.log"

for attempt in $(seq 1 20); do
	MIOPEN_FIND_MODE=FAST PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$PWD" \
		"$PY" -m vall_e.v3.train --out "$out" "$@" >> "$out/run.log" 2>&1
	rc=$?  # before anything else runs: "$(date)" below would reset $?
	[ $rc -eq 0 ] && break
	echo "$(date) exited with $rc (139 = segfault), resuming (attempt $attempt)" >> "$out/run.log"
	sleep 30
done
echo "$(date) done" >> "$out/run.log"
