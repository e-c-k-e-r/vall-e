# `utils/*`

This folder contains helper utilities for either training or general functions of the program.

These scripts are to remain agnostic to any model, to allow for reuse for other applications.

## `utils/distributed.py`

This script contains the necessary code needed to utilize distributed training.

Attributions are noted at the top.

## `utils/io.py`

This script contains the necessary code for loading and storing state dicts, through pickles (`.pt`) or SafeTensors (`.sft`), and offers parity for each storage type.

Additionally, some JSON helper functions are provided here.

## `utils/pattern.py`

This script contains (unused) code related to formatting sequences of audio codes into different pattern types.

Attributions are noted at the top.

## `utils/sampler.py`

This script contains code to handle sampling from a list of indices.
* `PoolSampler` has a master list of indices "in the marble bag" that are sampled without replacement.
* `OrderedSampler` will output indices from 0 to `length`, in order.
* `BatchedOrderedSampler` does the above, but will output lists of indices instead.
* `RandomSampler` will output indices from 0 to `length`, randomly.

Each sampler can load and store a state dict.

## `utils/unsloth.py`

This script contains Unsloth, a VRAM-saving optimization that offloads the input tensors to CPU on a backwards pass.

This is mostly unncessary, as inputs are rather small themselves, but is offered nonetheless if needed through `cfg.optimizations.unsloth = True`

Attributions are noted at the top.

## `utils/utils.py`

This script contains additional helper functions that do not require a dedicated file.

## `utils/train.py`

This script handles the necessary code for training, such as:
* iterating through a dataloader
* iterating through an `Engines` to train each underlying `Engine`
* printing training metrics
* invoking `save`, `eval`, `export` every X iterations
* handling stdin commands, such as `save`, `export`, `eval`, and `quit`

## `utils/wrapper.py`

This script contains optimizations and additional code that require injecting or replacing modules.

Most configurations are offered through `cfg.optimization`.