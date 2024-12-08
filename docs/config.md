# `config.py`

This script handles everything related to storing configuration information, as well as:
* loading the `data.h5` file
* loading the phoneme tokenizer

Thorough documentation pertaining to each field should already be noted alongside each line, or in the [provided YAML](/data/config.yaml).

## `BaseConfig`

This serves as an agnostic base class that can be reused across additional projects.

Aside from accessing properties, the end user should not be required to interact with this.

## `Config`

This serves as the inhereted class for `BaseConfig`, which contains instances of the following classes within it.

Additional global-states can be found here, such as:
* `device`: which device to load the model to
* `experimental`: a debug flag
	* for the end user, this gates off experimental sampler settings in the web UI.
* `tokenizer`: the tokenizer type to use
	* this only really is used for the `ar+nar-retnet-8`, as it used a naive tokenizer and vocab.
* `tokenizer_path`: the path to the tokenizer's vocab to use
	* this should be left alone for the end user.
* `audio_backend`: which audio backend to use.
	* supported options are `encodec`, `vocos`, and `dac`.
	* the end user should not touch this, as this not only depends on the model used, but also governs what audio codec to store processed audio under for the dataset.
* `weights_format`: the default weights format to save and load state dicts to
	* the end user shouldn't worry about this, as SafeTensors are primarily used, but the program can easily handle any pickled dicts if requested.
* `weights_name`: the name (without the extension) to load the weights from directly. Defaults to `fp32`.
	* the end user shouldn't worry about this, but it makes regression testing much easier without needing to juggle renaming files.

On initialization, this class then validates its member variables to ensure they're instances of the below classes, rather than dicts.
* Backwards compatibility validation may be performed during this step as well.
* The tokenizer and HDF5 dataset (if requested) is instantiated and initialized here too.

## `Dataset`

This class contains configuration options pertaining to the dataset and dataloader for the program, as documented under [/docs/data.md](/docs/data.md).

This is *mostly* agnostic, but VALL-E specific options can easily be gutted.

## `Model`

This class contains configuration options pertaining to a loaded model, both model specifications and model-specific runtime options (such as the attention mechanism).

This can be stored alongside a state dict to allow for loading stored weights directly without need for a config YAML.

This is *mostly* agnostic, but VALL-E specific options can easily be gutted.

### `ModelExperimentalSettings`

This class contains experimental knobs and dials that offer zero guarantees that modify model, training, or inferencing behavior.

The end user should *not* mess with these unless you know what you're doing, as output will greatly vary.

## `LoRA`

Similar to `Model`, this stores settings pertaining to the LoRA(s) to load for training or inferencing.

Like `Model`, these settings can also be stored alongside a LoRA's state dict to be loaded directly without need for a config YAML.

## `Hyperparameters`

This class defines the hyperparameters to use during training.

For the most part, when using `prodigyopt`, the only dials to care about is `batch_size` and `gradient_accumulation_step`.

For knowledge distillation, its corresponding hyperparameters live here, rather than alongside a given model's configuration.

## `Evaluation`

This class governs the behavior during the evaluation / validation pass during training.

If `cfg.evaluation.size > 0`, then the evaluation / validation passes are triggered every `cfg.evaluation.frequency` iteration steps.

During evaluation:
* for the `subtrain` evaluation pass, the training dataset is directly sampled through indices, rather than the iterator, to avoid having to duplicate the dataset.
	* in the future, the samples during this pass should sample around the training dataloader's current position.
* for the `val` validation pass, the validation dataset is sampled through the dataloader's iterator.
	* currently, the validation dataloader's sampler is not stored.

A total of `cfg.evaluation.size` samples are inferenced in no more than `cfg.evaluation.batch_size`-sized batches (no more than, because batched samplers may return different sized batches).

The resulting audio is then stored within the current log directory (`./{YAML_PATH}/logs/{START_TIME}/{CURRENT_ITERATION}/`), storing the input audio prompt, the resulting output, and the target output.

The resultant waveform compared against the target waveform using AuraLoss's `MelSTFTLoss` to compare similarities, and the loss is logged.
* To-do: replace this with a better method.

The inference settings used for the evaluation / validation pass can be defined under `cfg.evaluation.kwargs`, where each entry should mirror the CLI arguments for inferencing.

## `Trainer`

This class governs the trainer's behavior during training, from:
* which checkpoint to save and load from
* when loading the state dict or checkpoint
* when to save (or export) every X iterations
* what to do when an OOM error is caught, if it should catch those thrown exceptions
* which `Engine` backend to use
* what data type to load the model for training under, and to use mixed precision

### `DeepSpeed`

This class handles the config dict that is passed to DeepSpeed for initialization.

DeepSpeed-specific features like "compression training" (which for the purpose of VALL-E is superfluous) and use of ZeRO (which for the purpose of VALL-E is only really needed if you're on very low VRAM for training).

The dict can be overriden under `cfg.trainer.deepspeed.config`, to explicitly provide options.

## `Inference`

This class handles inferencing behavior, such as:
* which `Engine` backend to use
* what data type to load the model for inferencing under, and to use mixed precision

## `Optimizations`

This class handles enabling requested optimization techniques and frameworks, such as:
* BitsAndBytes
* DAdaptation
* BitNet
* Nvidia's TPE's FP8
* Unsloth input tensor offloading

as well as modifying how optimization techniques and frameworks, by either replacing the original module within the model, or by injecting the optimized version of the model over the original model.
* In other words, `replace` will not override the original classes under torch, while `inject` is a more invasive method.
* For all intents and purposes, use `replace`.

Additionally, an experimental method of offloading the model between different devices can be done through `model_offloading`.
* However, this feature needs validation, as this was partially tested forever ago.

---

## `NaiveTokenizer`

This is a simple class that handles tokenizing from my original, naive way. The `ar+nar-retnet-8` uses this form of tokenizing, which simply mainly does some funny string manipulation to handle token merges.

The reference model `ar+nar-llama-8` *could* use this, but for how reliant it is on the remaining tokens in the vocab being merges, requires better merging logic.