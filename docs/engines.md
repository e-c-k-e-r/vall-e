# `engines/*`

This folder contains the necessary abstractions for handling training of models through either a local (`base`) backend, or additional wrappers (like DeepSpeed, and in the future Accelerate and Lightning).

This architecture is partially lifted from the original implementation, but expanded for both my needs and modularity for other backends.

An `Engine` is just a wrapper that contains training metadata for the loaded module.

An `Engines` is a dict of `Engine`s, and extra functions to allow iterating through its contents, allowing for simultaneous loading and training of engines for a shared dataloader iteration.

## `__init__.py`

This script handles the bulk of loading a model and wrapping the model with the requested engine type.

The checkpoint or weight path is automatically deduced, as well as pre-processing the state dict (if requested) before loading it.
* resizing modules from the weights to the requested configuration in the YAML is done here.
* replacing modules with optimized versions or LoRAs are applied here.
* the requested optimizer, and params to freeze, for a model is applied here.

## `base.py`

The internal (`local`) implementation of orchestrating training. The basics are handled here, from automatic-mixed-precision, gradient accumulation, loss scaling, etc.

Functions for other backends are also defined here, such as the training step function.

## `deepspeed.py`

A backend relying on `deepspeed` for its orchestration, which offers additional features that can be defined under `cfg.trainer.deepspeed`.