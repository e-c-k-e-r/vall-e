# `data.py`

This script handles the meat of preparing the data to feed the model through the dataloader, and unfortunately makes up for quite a lot of this project's complexity.

Most of these settings live under `cfg.dataset`.

## Dataset

The provided reference model was trained on `?`k hours of audio with a mix of:
* 490.151 hours (out of 585 hours) of LibriTTS-R's entire dataset
* 8362.304 hours (out of 10270 hours) of `small`+`medium`+`duplicate` portions of LibriLight
* 4467.611 hours (out of `?` hours) of Emilia's German, French, and Japanese dataset
* 2927.186 hours (out of `?` hours) of a privately sourced corpus of 425 audiobooks
* 2364.799 hours (out of `?` hours) of Emilia's English dataset
* 54.775 hours of a personal small corpus of transcribed utterances from a selection of video games

These durations were reported from the training script directly.
* Utterances under 3 seconds or above 32 seconds were culled from the duration count.
* Metadata was *mostly* derived from the transcription metadata, mostly.
	* LibriTTS-R's duration metadata was derived from the quantized audio size.

### Leverage Your Own Dataset

If you already have a dataset you want, for example, your own large corpus or for finetuning, you can use your own dataset instead.

1. Populate your source voices under `./voices/{group name}/{speaker name}/`.

2. Run `python3 -m vall_e.emb.transcribe`. This will generate a transcription with timestamps for your dataset.
  + If you're interested in using a different model, edit the script's `model_name` and `batch_size` variables.

3. Run `python3 -m vall_e.emb.process`. This will phonemize the transcriptions and quantize the audio.
  + If you're using a Descript-Audio-Codec based model, ensure to set the sample rate and audio backend accordingly.

4. Run `python3 -m vall_e.emb.similar`. This will calculate the top-k most similar utterances for each utterance for use with sampling.
  + Doing this will help the model follow the input prompt stronger, at the possible "cost" of the model not learning how to "infer" the target speaker AND prosidy.

5. Copy `./data/config.yaml` to `./training/config.yaml`. Customize the training configuration and populate your `dataset.training` list with the values stored under `./training/dataset/list.json`.
  + Refer to `./vall_e/config.py` for additional configuration details.

### Dataset Formats

Two dataset formats are supported:
* the standard way:
  - data is stored under `./training/data/{group}/{speaker}/{id}.{enc|dac}` as a NumPy file, where `enc` is for the EnCodec/Vocos backend, and `dac` for the Descript-Audio-Codec backend.
  - it is *highly* recommended to generate metadata to speed up dataset pre-load with `python3 -m vall_e.data --yaml="./training/config.yaml" --action=metadata`
* using an HDF5 dataset:
  - you can convert from the standard way with the following command: `python3 -m vall_e.data --yaml="./training/config.yaml"` (metadata for dataset pre-load is generated alongside HDF5 creation)
  - this will shove everything into a single HDF5 file and store some metadata alongside (for now, the symbol map generated, and text/audio lengths)
  - be sure to also define `use_hdf5` in your config YAML.

## Dataloader

The dataloader handles some simple yet effective features, such as:
* culling samples within a requested duration range
* grouping samples based on:
	* speakers (to keep utterances for a given speaker) and groups (to keep similar speakers within a group as defined in the dataset)
	* durations, to keep VRAM usage and throughput consistent, if requested (as training requires keeping *all* samples of a batch the same token length)
* further partitioning samples per GPU
* shuffling then interleaving, per the dataloader sampler settings
* saving/loading sampler states to disk
* preparing a sample in a batch with adequate data for a given task, such as:
	* picking an input prompt similar to the sampled output audio, if requested
	* picking an input prompt from the same speaker as the sample, if the above is not requested
	* preparing the input sequence for the given task (such as non-TTS tasks)

If `cfg.dataset.cache == True`, the initial list of paths and duration metadata (used for sorting/bucketing) is cached ~~through `diskcache`~~ under `{YAML_PATH}/.cache/{DATASET_HASH}/`. To allow for seamless modifications to the loaded dataset, the `DATASET_HASH` relies on:
* duration range
* folders/groups in the dataset
* if using HDF5 (due to the key format differing)

Be sure to delete the resultant `.cache` folder, as well as the `sampler.*` state dicts alongside checkpoints, if you plan to modify the dataloader settings between training sessions.

## Tasks

As this handles preparing the data fed into the model for training, this script needs to be aware of what tasks it should attend to, as mostly outlined under SpeechX.

This section may be covered elsewhere in the documentation, but coverage here should focus on the specifics of attending to the task, rather than what the task is.

* `tts`: basic and naive text-to-speech.
	* requires a text transcription, input audio prompt, and the output audio response.
* `tts-c`: also noted as "VALL-E Continuous"
	* this is what most other TTS solutions abide by (those that require a transcription of the input prompt)
	* this *should* be more accurate as it has the output adhere stronger to the input through guidance, but doesn't seem to be necessary (to train for or inference under).
	* naively, this requires just the text transcription and output audio response, where part of the output audio response is trimmed to serve as the input audio prompt.
	* non-naively, this requires two text transcriptions, and two output audio responses (where one of them serve as the input audio prompt).
* `stt`: basic and naive speech-to-text.
	* requires an input audio prompt and the output text transcription (as phonemes, unfortunately).
* `ns`: noise suppression.
	* requires just a text transcription and an output audio response, where the input audio prompt is just the output + noise
	* text transcription can optionally be removed to allow for training without text guidance.
* `sr`: speech removal.
	* requires just a text transcription and an output audio response, where the input audio prompt is just the sampled utterance + noise, and the output is just the original noise.
	* text transcription can optionally be removed to allow for training without text guidance.
* `tse`: target speech extraction.
	* requires a text transcription, an input audio prompt of the sampled speaker, utterance sampled from a different speaker, and the output audio response.
	* the input prompt is appended with both the output audio and the utterance sampled from a different speaker overlayed on one another.
* `cse`: clean speech editing.
	* an ideal world would  have phoneme-level transcriptions, but I do not have very-accurate phoneme-level transcriptions.
	* to make up for this, this requires multiple samples for the prefix, the original middle, the edited portion for the middle, and the postfix sample.
		* the prefix and postfix *can* be randomly omitted, but keeping them in ensures better editing of speech within the middle.
	* requires four full samples.
* `nse`: noisy speech editing.
	* the above, but injects some noise throughout the sampled utterances.

A mystical `vc` for performing voice conversion is possible, but either requires a dataset to do so, or abusing an emergent property.
* This emergent property is mostly abused through the NAR-len's demasking routine.

## `__main__`

This script can be called directly to perform dataloader-related tasks.

### `--action=metadata`

Invoking this will take processed samples (`.enc` for EnCodec, `.dac` for Descript-Audio-Codec) from `{YAML_PATH}/data/`, as per the YAML's `cfg.dataset.{training|validation|noise}` lists, and store helpful metadata under `{YAML_PATH}/metadata/`, to speed up dataloader preparations. Since dataloader preparations can cull based on audio durations, being able to look up a sample's duration speeds things up without needing to load the sample and read the file's metadata.

This metadata can be then used to store similar speaker indices.

### `--action=hdf5`

Invoking this will take processed samples (`.enc` for EnCodec, `.dac` for Descript-Audio-Codec) from `{YAML_PATH}/data/`, as per the YAML's `cfg.dataset.{training|validation|noise}` lists, and store them within a single `.h5` HDF5 file.

Additionally, this implicitly invokes `--action=metadata`, to create additional JSON metadata under `{YAML_PATH}/metadata/`, to speed up dataloader preparations.

### `--action=sample`

Invoking this will load the dataloader, sample it, and print out the batch's contents.

This serves primarily for debugging purposes during development, and should not be necessary for the end user.

### `--action=validate`

Invoking this will process the dataset to check for any phonemes missing from the tokenizer (as defined under `cfg.tokenizer`).

Any missing phonemes will be printed through `logger` to make mending the tokenizer dict easier.

This serves primarily for debugging purposes during development, and should not be necessary for the end user. However, additional languages may emit additional IPAs through `phonemizer`, so those training additional languages should take care to validate for missing phonemes before training, to avoid headaches.

## `cfg.dataset`

This entry in the config YAML handles knobs and features related to the dataloader. This is defined as `Dataset` in `./vall_e/config.py`
* `training`: list of entries to populate the training dataset with. Wildcards are accepted, such as `LibriVox/*` to easily load a speaker within a group, without needing to define them individually.
* `validation`: the above, but for the validation dataset.
* `noise`: the above, but for any noise that may be sampled during dataloader sampling. Text is not required for this dataset.
* `speaker_name_getter`: a lambda function to evaluate, to retrieve the speaker name from a given path string.
* `speaker_group_getter`: a lambda function to evaluate, to retrieve the speaker's associated group from a given path string.
* `speaker_languages`: Deprecated. This is a dict that maps language codes to a list of speaker groups, for when the language code was not stored alongside a sample's data.
* `use_hdf5`: use `{YAML_PATH}/{cfg.dataset.hdf5_name}` to sample data from, rather than individual files on disk.
* `hdf5_name`: filename (or path?) to the HDF5 dataset file to load, if the above is requested.
* `hdf5_flag`: flag to open the above HDF5 file under. By default this is `a` to write to, as it's necessary for HDF5 creation, but will automatically set to `r` under distributed settings.
* `use_metadata`: references generated metadata instead of loading samples individually to acquire metadata.
* `validate`: cull samples that do not fall within the requested `cfg.dataset.duration_range`.
* `workers`: number of worker processes to handle dataloading under PyTorch.
* `cache`: use diskcache when requested to not require subsequent processing. This handles *all* `diskcache` requests throughout the program if requested, but should only really be used under this script.
* `min_utterances`: number of utterances to treat a speaker as valid.
* `max_utterances`: maximum number of utterances a speaker can have. The remaining utterances are sliced off.
	* This is beneficial if you happen to have a section of your dataset with a ton of speakers, but you want to train on a plethora of speakers instead to balance out speaker.
* `duration_range`: a list of two values to denote the acceptable duration ranges a sample is valid for the dataloader. 
* `sample_type`: type of sampler to use. Currently accepts `path` (an epoch is all paths in the dataset, and each index maps to each sample) or `speaker` (an epoch is all speakers in the dataset, and each index maps to each speaker)
* `sample_order`: order to keep the dataloader sample. Currently accepts `interleaved` (tries to balance per speaker) and `duration` (orders by duration to keep throughput and VRAM usage consistent).
* `sample_shuffle`: shuffles the dataloader sampler.
* `sample_max_duration_batch`: the maximum total duration a batch can be. Values > 0 will enable batch sampling, where the dataloader sampler returns batches of batches.
	* This only works under `sample_order=duration` and `sample_type=path`, and should raise an exception for any other configuration.
* `prompt_duration_range`: a list of two values to denote the range a sample's input prompt should be. This keeps the model trained for input prompt durations within these, and a little extra sometimes works without training for it.
* `prompt_max_samples`: maximum number of utterances to sample for an input prompt to combine, if needed to fill the above duration window.
* `prompt_continuous_utterance_p`: probability for a sample's input prompt to instead be the output prompt, and prepare the sample under "continuous" mode.
* `prompt_similar_p`: probability to use a sample's most similar utterance as the input prompt, rather than randomly picking another utterance of the same speaker.
	* This requires adequate metadata to be available to store the top-K similar indices.
* `prompt_similar_top_k`: use the top-k candidates for the above sampling.
* `prompt_similar_top_k_offset`: the above, but an offset (as in it will not use the top-K-offset most similar utterances).
* `prompt_inject_noise`: inject some noise in a sample's input prompt. *Will* harm dataloader throughput, as it requires re-encoding the audio.
* `resps_max_samples`: maximum utterances to use for the sample's input text and output response audio.
* `resps_append_p`: probability to append additional utterances to the sample.
* `resps_pad_silence_p`: probability to pad the output response audio with silence. Does *not* require re-encoding, unless requested through `reencode_on_concat`.
* `tasks_list`: list of task names a sample can be.
	* Currently supports: `tts`, `stt`, `tts-c`, `ns`, `sr`, `tse`, `nse`, `cse`
* `reencode_on_concat`: if enabled, audio will be decoded to a raw waveform, concatted, then reencoded, instead of naively concatting EnCodec codes.
	* This isn't necessary naively concatting offers trivial inaccuracies.
* `reencode_device`: device to load EnCodec within the dataloader.
	* *technically* only `cpu` should be supported, as loading models in dataloaders causes problems?
* `noise_scale`: multiplier to the noise when applying noise. Lower numbers keep it quieter.
* `retokenize_text`: if the text/phoneme transcription is available in the metadata, use that to re-tokenize instead of relying on the stored tokens itself.
	* This is helpful if you modify the tokenizer dict in post, but do not want to re-process the dataset to modify the tokenized phonemes.
* `_frames_per_second`: overrides the internal tokens-per-second-of-audio ratio. Should never require modifying.