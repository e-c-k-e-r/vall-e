# `plot.py`

Included is a helper script to parse the training metrics. Simply invoke it with, for example: `python3 -m vall_e.plot --yaml="./training/config.yaml"`

You can specify what X and Y labels you want to plot against by passing `--xs tokens_processed --ys loss.nll stats.acc`