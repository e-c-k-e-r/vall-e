# todo: clean this mess up

from .config import cfg
from .data import create_train_val_dataloader
from .emb import qnt

from .utils import setup_logging, to_device, trainer, flatten_dict, do_gc
from .data import fold_inputs, unfold_outputs
from .utils.distributed import is_global_leader

import auraloss
import json
import logging
import random
import torch
import torch.nn.functional as F
import traceback
import shutil

from collections import defaultdict

from tqdm import tqdm
import argparse

_logger = logging.getLogger(__name__)

mel_stft_loss = auraloss.freq.MelSTFTLoss(cfg.sample_rate, device="cpu")

def train_feeder(engine, batch):
	with torch.autocast("cuda", dtype=cfg.trainer.dtype, enabled=cfg.trainer.amp):
		batch_size = len(batch["text"])
		engine.current_batch_size = batch_size

		if engine.hyper_config.experimental.hf:
			if engine.hyper_config.experimental.interleave:
				quant_levels = 0	
				resps_list = [ resp for resp in batch["resps"] ]
			else:
				quant_levels = [ random.randint( 0 if "ar" in cfg.model.capabilities else 1, cfg.model.max_levels) for _ in range(batch_size) ]
				resps_list = [ [] if l == 0 else resp for l, resp in zip(quant_levels, batch["resps"]) ]

			input_ids, attention_mask = fold_inputs(
				text_list=batch["text"],
				prom_list=batch["proms"],
				resp_list=resps_list,
				targ_list=batch["resps"],
				quant_levels=quant_levels,
			)
			target_ids, target_attention_mask = fold_inputs(
				text_list=batch["text"],
				prom_list=batch["proms"],
				resp_list=resps_list,
				targ_list=batch["resps"],
				quant_levels=quant_levels,
				ignore_index=-100
			)
			engine(
				input_ids=input_ids,
				labels=target_ids,
			)
		else:
			engine(
				text_list=batch["text"],
				proms_list=batch["proms"],
				resps_list=batch["resps"],
				lang_list=batch["lang"],

				training=True,
			)

		losses = engine.gather_attribute("loss")
		stat = engine.gather_attribute("stats")

		loss = torch.stack([*losses.values()]).sum()

	stats = {}
	stats |= {k: v.item() for k, v in losses.items()}
	stats |= {k: v.item() for k, v in stat.items()}

	engine.tokens_processed += sum([ text.shape[0] for text in batch["text"] ])
	engine.tokens_processed += sum([ resps.shape[0] for resps in batch["resps"] ])

	return loss, stats

@torch.inference_mode()
def run_eval(engines, eval_name, dl):
	stats = defaultdict(list)
	stats['loss'] = []

	def process( name, batch, resps_list ):
		for speaker, path, ref, hyp, prom, task in zip(batch["spkr_name"], batch["path"], batch["resps"], resps_list, batch["proms"], batch["task"]):
			if len(hyp) == 0:
				continue

			filename = f'{speaker}_{path.parts[-1]}'

			if task != "tts":
				filename = f"{filename}_{task}"

			# to-do, refine the output dir to be sane-er
			ref_path = (cfg.log_dir / str(engines.global_step) / "ref" / filename).with_suffix(".wav")
			hyp_path = (cfg.log_dir / str(engines.global_step) / name / eval_name / filename).with_suffix(".wav")
			prom_path = (cfg.log_dir / str(engines.global_step) / name / "prom" / filename).with_suffix(".wav")

			hyp_path.parent.mkdir(parents=True, exist_ok=True)
			ref_path.parent.mkdir(parents=True, exist_ok=True)
			prom_path.parent.mkdir(parents=True, exist_ok=True)
			
			ref_audio, sr = qnt.decode_to_file(ref, ref_path)
			hyp_audio, sr = qnt.decode_to_file(hyp, hyp_path)
			prom_audio, sr = qnt.decode_to_file(prom, prom_path)

			# pseudo loss calculation since we don't get the logits during eval
			min_length = min( ref_audio.shape[-1], hyp_audio.shape[-1] )
			ref_audio = ref_audio[..., 0:min_length]
			hyp_audio = hyp_audio[..., 0:min_length]
			stats['loss'].append(mel_stft_loss(hyp_audio[None, :, :], ref_audio[None, :, :]).item())
	
	processed = 0
	while processed < cfg.evaluation.size:
		batch = to_device(next(iter(dl)), cfg.device)

		# limit to eval batch size in the event we somehow have a weird dataloader
		for key in batch.keys():
			batch[key] = batch[key][:cfg.evaluation.batch_size]

		processed += len(batch["text"])

		for name in engines:
			engine = engines[name]

			if engine.hyper_config.experimental.hf:
				if engine.hyper_config.experimental.interleave:
					input_ids, attention_mask = fold_inputs(
						text_list=batch["text"],
						prom_list=batch["proms"],
					)
					output = engine.module.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=cfg.evaluation.steps, eos_token_id=3, do_sample=False)
					resps_list = unfold_outputs( output )["resp_list"]
				else:
					steps = cfg.evaluation.steps
					resps_list = [ [] for _ in range(len(text_list)) ]
					for l in range(cfg.model.max_levels):
						quant_levels = [ [ l ] for _ in range(len(text_list)) ]

						input_ids, attention_mask = fold_inputs(text_list=batch["text"], prom_list=batch["proms"], resp_list=resps_list, quant_levels=quant_levels, experimental=True)
						min_length = 1 
						for batch in input_ids:
							min_length = max( min_length, batch.shape[0] + 1 )

						output = model.generate(
							input_ids=input_ids,
							attention_mask=attention_mask,
							min_length=min_length,
							max_length=min_length+steps*(2 if l > 0 else 1),
							eos_token_id=3,
							do_sample=False
						)
						
						unfolded = unfold_outputs( output, quant_levels=quant_levels )

						if l == 0:
							steps = 0

						for batch, resp in enumerate(unfolded["resp_list"]):
							length = resp.shape[-1]

							# store length
							if l == 0:
								steps = max( steps, length )
							# pad
							else:
								resp = resp[:steps]
								if length < steps:
									resp = torch.cat([ resp, torch.Tensor([ 0 for _ in range(steps-length) ]).to(resp) ])

							resps_list[batch].append( resp )

					for i, resp in enumerate( resps_list ):
						resps_list[i] = torch.stack( resp ).t()
			else:
				if "len" in engine.hyper_config.capabilities:
					len_list = engine(text_list=batch["text"], proms_list=batch["proms"], max_steps=10 ) # don't need more than that
					resps_list = engine( text_list=batch["text"], proms_list=batch["proms"], len_list=len_list )
				else:
					if "ar" in engine.hyper_config.capabilities:
						resps_list = engine(text_list=batch["text"], proms_list=batch["proms"], lang_list=batch["lang"], max_steps=cfg.evaluation.steps, sampling_temperature=cfg.evaluation.ar_temperature)
					else:
						resps_list = [ resp[:, 0] for resp in batch["resps"] ]

					if "nar" in engine.hyper_config.capabilities:
						resps_list = engine(text_list=batch["text"], proms_list=batch["proms"], lang_list=batch["lang"], resps_list=resps_list, sampling_temperature=cfg.evaluation.nar_temperature)

			process( name, batch, resps_list )


	stats = {k: sum(v) / len(v) for k, v in stats.items()}
	engines_stats = {
		f'{name}.{eval_name}': stats,
		"it": engines.global_step,
	}
	#engines_stats['epoch'] = iteration * cfg.hyperparameters.gradient_accumulation_steps / len(dl)

	if cfg.trainer.no_logger:
		tqdm.write(f"Validation Metrics: {json.dumps(engines_stats)}.")
	else:
		_logger.info(f"Validation Metrics: {json.dumps(engines_stats)}.")


def train():
	parser = argparse.ArgumentParser("VALL-E TTS")
	parser.add_argument("--eval", action="store_true", default=None)
	args, unknown = parser.parse_known_args()

	# create log folder
	setup_logging(cfg.log_dir)
	# copy config yaml to backup
	if cfg.yaml_path is not None and is_global_leader():
		shutil.copy( cfg.yaml_path, cfg.log_dir / "config.yaml" )

	train_dl, subtrain_dl, val_dl = create_train_val_dataloader()
	
	def eval_fn(engines):
		do_gc()
		engines.eval()
		# wrapped in a try block because it's sometimes prone to breaking
		try:
			run_eval(engines, "subtrain", subtrain_dl)
			run_eval(engines, "val", val_dl)
		except Exception as e:
			print("Error occurred while performing eval:", str(e))
			print(traceback.format_exc())

		engines.train()
		qnt.unload_model()
		do_gc()

	qnt.unload_model()

	if args.eval:
		return eval_fn(engines=trainer.load_engines())

	"""
	if cfg.trainer.load_webui:
		from .webui import start
		start(lock=False)
	"""

	trainer.train(
		train_dl=train_dl,
		train_feeder=train_feeder,
		eval_fn=eval_fn,
	)

if __name__ == "__main__":
	# to-do: for DDP, spawn multiprocess instead of requiring `torchrun --nnodes=1 --nproc-per-node=4 -m vall_e.train yaml="./data/config.yaml"`
	train()
