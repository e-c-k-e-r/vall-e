#!/bin/bash

python3 -m venv venv
pip3 install -e .

mkdir -p ./training/valle/ckpt/ar+nar-retnet-8/
wget -P ./training/valle/ckpt/ar+nar-retnet-8/ "https://huggingface.co/ecker/vall-e/resolve/main/ckpt/ar%2Bnar-retnet-8/fp32.pth"
wget -P ./training/valle/ "https://huggingface.co/ecker/vall-e/raw/main/config.yaml"
