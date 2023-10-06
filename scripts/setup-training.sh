#!/bin/bash
`dirname $0`/setup.sh

wget -P ./training/valle/ "https://huggingface.co/ecker/vall-e/resolve/main/data.tar.gz"
wget -P ./training/valle/ "https://huggingface.co/ecker/vall-e/resolve/main/.cache.tar.gz"
tar -xzf ./training/valle/data.tar.gz -C "./training/valle/" data.h5
tar -xzf ./training/valle/.cache.tar.gz -C "./training/valle/"