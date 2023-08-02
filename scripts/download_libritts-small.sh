#!/bin/bash

# do not invoke directly in scripts
if [[ ${PWD##*/} == 'scripts' ]]; then
	cd ..
fi

# download training data
git clone https://huggingface.co/datasets/ecker/libritts-small ./data/libritts-small