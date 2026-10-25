#!/bin/bash
# After the cumulative ablations (so every rung sees the same store): adds the Emilia/DE utterances the first
# conversions missed (artifacts split between Emilia/DE/<speaker>/ and Emilia/DE/Emilia-DE/<speaker>/), then rebuilds the similar and exclude
# sidecars, which are sized to the store.
# usage: nohup scripts/v3-store-recover2.sh >> training/v3-store-recover.log 2>&1 &
cd "$(dirname "$0")/.."
S=data/store
until grep -q "cumulative ablations done" training/v3-after-main5.log 2>/dev/null; do sleep 120; done
while ps -eo args | grep -v grep | grep -q "vall_e.v3.train"; do sleep 60; done
echo "$(date) store before: $(python3 -c "import json; m=json.load(open('$S/manifest.json')); print(sum(s['utterances'] for s in m['shards']), 'utterances,', len(m['shards']), 'shards')")"
PY=python
PYTHONPATH=$PWD $PY -m vall_e.v3.store convert --meta training/llama-nemo/metadata --data training/llama-nemo/data --out $S \
	--groups Emilia/DE --supplement --workers 6 || exit 1
echo "$(date) store after: $(python3 -c "import json; m=json.load(open('$S/manifest.json')); print(sum(s['utterances'] for s in m['shards']), 'utterances,', len(m['shards']), 'shards')")"
PYTHONPATH=$PWD $PY -m vall_e.v3.store similar --store $S --meta training/llama-nemo/metadata --workers 6 || exit 1
PYTHONPATH=$PWD $PY -m vall_e.v3.store exclude --store $S || exit 1
echo "$(date) store recovery done"
