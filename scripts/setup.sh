#!/bin/bash

python3 -m venv venv
pip3 install -e .

mkdir -p ./training/valle/ckpt/ar+nar-retnet-8/
wget -P ./training/valle/ckpt/ar+nar-retnet-8/fp32.pth "https://huggingface.co/ecker/vall-e/resolve/main/ckpt/ar%2Bnar-retnet-8/fp32.pth"
wget -P ./training/valle/data.h5 "https://huggingface.co/ecker/vall-e/resolve/main/data.h5"
wget -P ./training/valle/config.yaml "https://huggingface.co/ecker/vall-e/raw/main/config.yaml"