#!/bin/bash

# do not invoke directly in scripts
if [[ ${PWD##*/} == 'scripts' ]]; then
	cd ..
fi

# download training data
cd data
mkdir librilight-tts
if [ ! -e ./librispeech_finetuning.tgz ]; then
	wget https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz
fi
tar -xzf librispeech_finetuning.tgz
cd ..

# clean it up
python3 ./scripts/prepare_libri.py

# convert to wav
pip3 install AudioConverter
audioconvert convert ./data/librilight-tts/ ./data/librilight-tts --output-format .wav

# process data
ulimit -Sn `ulimit -Hn` # ROCm is a bitch
python3 -m vall_e.emb.g2p ./data/librilight-tts # phonemizes anything that might have been amiss in the phoneme transcription
python3 -m vall_e.emb.qnt ./data/librilight-tts