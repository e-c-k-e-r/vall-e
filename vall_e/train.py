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

def train_feeder(engine, batch, teacher=None):
	engine.tokens_processed += sum([ text.shape[0] for text in batch["phns"] ])
	engine.tokens_processed += sum([ resps.shape[0] for resps in batch["resps"] ])

	with torch.autocast("cuda", dtype=cfg.trainer.dtype, enabled=cfg.trainer.amp):
		batch_size = len(batch["phns"])
		engine.current_batch_size = batch_size

		output = engine(
			phns_list=batch["phns"],
			proms_list=batch["proms"],
			resps_list=batch["resps"],
			lang_list=batch["lang"],
			tone_list=batch["tone"],
			task_list=batch["task"],
			text_list=batch["text"],

			training=True,
		)

		# get soft targets from teacher
		"""
		if teacher is not None:
			# extract inputs forwarded to model
			inputs = output.inputs

			# grab the teacher's logits
			with torch.no_grad():
				teacher_output = teacher.forward_super(
					inputs=inputs,
				)

			# KD hyperparameters
			T = cfg.hyperparameters.teacher_temperature
			A = cfg.hyperparameters.teacher_alpha
			L = cfg.hyperparameters.teacher_loss_fn

			# determine the output length for each batch (because blah blah some embeddings don't map to a discrete token anyways)
			# we could recreate the target sequence with the ignore indices put in, but that's agony
			student_logits = [ logit / T for logit in output.logits ]
			teacher_logits = [ logit / T for logit in teacher_output.logits ]

			if engine.module.ignore_inputs_for_loss:
				task_outputs = {
					"tts": "resp",
					"stt": "text",
					"len": "len",
				}
				output_lens = [ 0 for _ in range(batch_size) ]
				for batch_index, _batch in enumerate(inputs):
					task_type = "tts"
					for name, input in _batch:
						if name == "task":
							task_type = input

					for name, input in _batch:
						if name == task_outputs.get(task_type, name):
							output_lens[batch_index] = input.shape[0]

				# create probability distributions (literature says to have the students already log'd but not the teacher)
				student_logits = [ logit[-l:] for logit, l in zip( student_logits, output_lens ) ]
				teacher_logits = [ logit[-l:] for logit, l in zip( teacher_logits, output_lens ) ]

			if L == "kl":
				student_probs = [ F.log_softmax( logit, dim=-1 ) for logit in student_logits ]
				teacher_probs = [ F.log_softmax( logit, dim=-1 ) for logit in teacher_logits ]

				soft_losses = [ F.kl_div( student, teacher, reduction='batchmean', log_target=True ) for student, teacher in zip( student_probs, teacher_probs ) ]
			elif L == "mse":				
				soft_losses = [ F.mse_loss( student, teacher ) for student, teacher in zip( student_logits, teacher_logits ) ]

			for k in engine.module.loss.keys():
				engine.module.loss[k] *= (1.0 - A)
			engine.module.loss[L] = torch.stack(soft_losses).sum() * A * (T ** 2) / batch_size
		"""

		losses = engine.gather_attribute("loss")
		stat = engine.gather_attribute("stats")

		loss = torch.stack([*losses.values()]).sum()

	stats = {}
	stats |= {k: v.item() for k, v in losses.items()}
	stats |= {k: v.item() for k, v in stat.items()}

	return loss, stats

@torch.inference_mode()
def run_eval(engines, eval_name, dl, args=None):
	stats = defaultdict(list)
	stats['loss'] = []

	if cfg.evaluation.size == 0:
		return

	def process( name, batch, resps_list ):
		for speaker, path, ref, hyp, prom, task in zip(batch["speaker_name"], batch["path"], batch["resps"], resps_list, batch["proms"], batch["task"]):
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
		# directly randomly sample
		if eval_name == "subtrain":
			# sample from dataset
			# to-do: derive from current iteration
			samples = [ to_device(dl.dataset[random.randint( 0, len( dl.dataset ) )], cfg.device) for sample in range( cfg.evaluation.batch_size ) ]
			# collate manually
			batch = {k: [s[k] for s in samples] for k in samples[0]}
		else:
			batch = to_device(next(iter(dl)), cfg.device)

		# limit to eval batch size in the event we somehow have a weird dataloader
		for key in batch.keys():
			batch[key] = batch[key][:cfg.evaluation.batch_size]

		batch_size = len(batch["phns"])

		"""
		# to-do: eval for text tasks
		has_stt = False
		for i, task in enumerate( batch["task"] ):
			# easier to just change it to a tts task than drop stt tasks from the batch
			if task == "stt":
				# has_stt = True
				batch["task"][i] = "tts"
				batch["proms"][i] = batch["resps"][i][:75*3, :]
			elif task != "tts":
				batch["task"][i] = "tts"

		# random prompts requested
		if args and args.eval_random_text_prompts and eval_name == "subtrain":
			for i, _ in enumerate(batch["phns"]):
				batch["phns"][i] = get_random_prompt(tokenized=True).to(device=cfg.device)
				batch["resps"][i] = None
		"""

		processed += batch_size
		for name in engines:
			engine = engines[name]

			base_kwargs = dict(
				phns_list=batch["phns"],
				proms_list=batch["proms"],
				lang_list=batch["lang"],
				task_list=batch["task"],
				training=False,
			)

			with torch.autocast("cuda", dtype=cfg.trainer.dtype, enabled=cfg.trainer.amp):
				if engine.hyper_config.version >= 7:
					kwargs = base_kwargs | cfg.evaluation.kwargs
					# sample for NAR demask
					if random.random() < engine.hyper_config.experimental.masking_train_p:
						kwargs["len_list"] = [ resp.shape[0] for resp in batch["resps"] ]
					# inference
					resps_list = engine( **kwargs )
				else:
					if "len" in engine.hyper_config.capabilities:
						kwargs = base_kwargs | cfg.evaluation.kwargs
						max_steps = kwargs.pop("max_steps", 500)

						if "denoise_start" in kwargs:
							len_list = [ resp.shape[0] for resp in batch["resps"] ]
							kwargs["resps_list"] = [ resp[:, :1] for resp in batch["resps"] ]
						else:
							len_list = engine( max_steps=5, **kwargs )
							len_list = [ min( l, max_steps ) for l in len_list ]
						
						kwargs = base_kwargs | cfg.evaluation.kwargs
						resps_list = engine( **kwargs, len_list=len_list )
					else:
						if "ar" in engine.hyper_config.capabilities:
							kwargs = base_kwargs | cfg.evaluation.kwargs
							resps_list = engine( **kwargs )
						else:
							resps_list = [ resp[:, 0] for resp in batch["resps"] ]

						if "nar" in engine.hyper_config.capabilities:
							kwargs = base_kwargs | cfg.evaluation.kwargs
							resps_list = engine( **kwargs, resps_list=resps_list )

			process( name, batch, resps_list )

	stats = {k: sum(v) / len(v) for k, v in stats.items() if v}
	engines_stats = {
		eval_name: stats,
		"it": engines.global_step,
	}

	try:
		for name, engine in engines.items():
			if engine.wandb is not None:
				engine.wandb.log({
					f'{eval_name}.loss.mstft': stats['loss'],
				}, step=engine.global_step)
	except Exception as e:
		print(e)

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
	# create dataloaders
	train_dl, val_dl = create_train_val_dataloader()
	# evaluation lambda
	def eval_fn(engines):
		do_gc()
		engines.eval()
		# wrapped in a try block because it's sometimes prone to breaking
		try:
			run_eval(engines, "subtrain", train_dl, args)
			run_eval(engines, "val", val_dl, args)
		except Exception as e:
			_logger.warning(f"Error occurred while performing eval: {str(e)}")
			_logger.warning(traceback.format_exc())

		engines.train()
		qnt.unload_model()
		do_gc()
	# unload EnCodec if it's already loaded
	qnt.unload_model()
	# only eval is requested
	if args.eval:
		return eval_fn(engines=trainer.load_engines())

	"""
	# start web UI
	if cfg.trainer.load_webui:
		from .webui import start
		start(lock=False)
	"""
	# pre-training config validation
	if cfg.model.experimental.layerskip and cfg.trainer.weight_dtype == "float16":
		_logger.warning(f"Training with LayerSkip enabled with float16 may result in frying the model if the loss scale gets too small (<=8K) or with too large of a de facto batch size (>512 samples).")

	# train
	trainer.train(
		train_dl=train_dl,
		train_feeder=train_feeder,
		eval_fn=eval_fn,
	)

if __name__ == "__main__":
	# to-do: for DDP, spawn multiprocess instead of requiring `torchrun --nnodes=1 --nproc-per-node=4 -m vall_e.train yaml="./data/config.yaml"`
	train()
