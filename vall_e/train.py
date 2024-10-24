# todo: clean this mess up

from .config import cfg
from .data import create_train_val_dataloader, get_random_prompt, tokenize
from .emb import qnt, g2p

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

		engine(
			text_list=batch["text"],
			proms_list=batch["proms"],
			resps_list=batch["resps"],
			lang_list=batch["lang"],
			tone_list=batch["tone"],
			task_list=batch["task"],

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
def run_eval(engines, eval_name, dl, args=None):
	stats = defaultdict(list)
	stats['loss'] = []

	if cfg.evaluation.size == 0:
		return

	def process( name, batch, resps_list ):
		for speaker, path, ref, hyp, prom, task in zip(batch["spkr_name"], batch["path"], batch["resps"], resps_list, batch["proms"], batch["task"]):
			if len(hyp) == 0:
				continue

			filename = f'{speaker}_{path.parts[-1]}'

			if task != "tts":
				filename = f"{filename}_{task}"

			# flatten prom
			if not isinstance(prom, torch.Tensor) and prom is not None:
				prom = torch.concat([ p for p in prom if isinstance(p, torch.Tensor) ])

			# to-do, refine the output dir to be sane-er
			ref_path = (cfg.log_dir / str(engines.global_step) / "ref" / filename).with_suffix(".wav")
			hyp_path = (cfg.log_dir / str(engines.global_step) / name / eval_name / filename).with_suffix(".wav")
			prom_path = (cfg.log_dir / str(engines.global_step) / name / "prom" / filename).with_suffix(".wav")

			hyp_path.parent.mkdir(parents=True, exist_ok=True)
			ref_path.parent.mkdir(parents=True, exist_ok=True)
			prom_path.parent.mkdir(parents=True, exist_ok=True)
			
			hyp_audio, sr = qnt.decode_to_file(hyp, hyp_path)
			
			if ref is not None:
				ref_audio, sr = qnt.decode_to_file(ref, ref_path)

			if prom is not None:
				prom_audio, sr = qnt.decode_to_file(prom, prom_path)

			# naive loss calculation
			# to-do: find a better way to calculate this / a better metric
			if ref is not None:
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

		batch_size = len(batch["text"])

		# to-do: eval for text tasks
		has_stt = False
		for i, task in enumerate( batch["task"] ):
			# easier to just change it to a tts task than drop stt tasks from the batch
			if task == "stt":
				# has_stt = True
				batch["task"][i] = "tts"
				batch["proms"][i] = batch["resps"][i][:75*3, :]

		# random prompts requested
		if args and args.eval_random_text_prompts and eval_name == "subtrain":
			for i, _ in enumerate(batch["text"]):
				batch["text"][i] = get_random_prompt(tokenized=True).to(device=cfg.device)
				batch["resps"][i] = None

		processed += batch_size
		for name in engines:
			engine = engines[name]


			base_kwargs = dict(
				text_list=batch["text"],
				proms_list=batch["proms"],
				lang_list=batch["lang"],
				task_list=batch["task"],
			)

			if engine.hyper_config.experimental.hf:
				resps_list = engine( **base_kwargs )
			elif "len" in engine.hyper_config.capabilities:
				len_list = engine( **base_kwargs, max_steps=10 ) # don't need more than that
				len_list = [ min( l, cfg.evaluation.steps ) for l in len_list ]
				
				kwargs = base_kwargs | cfg.evaluation.nar_kwargs
				resps_list = engine( **kwargs, len_list=len_list )
			else:
				if "ar" in engine.hyper_config.capabilities:
					kwargs = base_kwargs | cfg.evaluation.ar_kwargs
					resps_list = engine( **kwargs )
				else:
					resps_list = [ resp[:, 0] for resp in batch["resps"] ]

				if "nar" in engine.hyper_config.capabilities:
					kwargs = base_kwargs | cfg.evaluation.nar_kwargs
					resps_list = engine( **kwargs, resps_list=resps_list )

			process( name, batch, resps_list )

			# evaluate why it's so slow
			if has_stt:
				max_steps = max( [ text.shape[0] for text in batch["text"] ] )

				kwargs["text_list"] = None
				kwargs["task_list"] = [ "stt" for _ in range(batch_size) ]
				kwargs["proms_list"] = [ ["stt"] for _ in range(batch_size) ]
				kwargs["resps_list"] = batch["resps"]

				text_list = engine( **kwargs, max_steps=max_steps, sampling_temperature=0.0)
				text_list = [ cfg.tokenizer.decode( text ) for i, text in enumerate( text_list ) ]

				_logger.info(f"Validation Metrics (STT): {text_list}")

	stats = {k: sum(v) / len(v) for k, v in stats.items() if v}
	engines_stats = {
		f'{name}.{eval_name}': stats,
		"it": engines.global_step,
	}
	#engines_stats['epoch'] = iteration * cfg.hyperparameters.gradient_accumulation_steps / len(dl)

	_logger.info(f"Validation Metrics: {json.dumps(engines_stats)}.")


def train():
	parser = argparse.ArgumentParser("VALL-E TTS")
	parser.add_argument("--eval", action="store_true", default=None)
	parser.add_argument("--eval-random-text-prompts", action="store_true", default=None)
	#parser.add_argument("--eval-random-audio-prompts", action="store_true", default=None)
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
			run_eval(engines, "subtrain", subtrain_dl, args)
			run_eval(engines, "val", val_dl, args)
		except Exception as e:
			_logger.warning(f"Error occurred while performing eval: {str(e)}")
			_logger.warning(traceback.format_exc())

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
