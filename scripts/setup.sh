#!/bin/bash

python3 -m venv venv
source ./venv/bin/activate
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121 # or cu118 / cu124
pip3 install -e .

mkdir -p ./training/valle/ckpt/ar+nar-llama-8/
wget -P ./training/valle/ckpt/ar+nar-llama-8/ "https://huggingface.co/ecker/vall-e/resolve/main/models/ckpt/ar%2Bnar-llama-8/fp32.sft"
wget -P ./training/valle/ "https://huggingface.co/ecker/vall-e/resolve/main/models/config.llama.yaml"
