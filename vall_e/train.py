# todo: clean this mess up
# todo: yank deepspeed dependent code out into its own thing

from .config import cfg
from .data import create_train_val_dataloader
from .emb import qnt

from .utils import setup_logging, to_device, trainer, flatten_dict, do_gc
from .utils import wrapper as ml

from .models import get_models

import auraloss
import deepspeed
import json
import logging
import random
import torch
import torch.nn.functional as F
import traceback

from collections import defaultdict

from deepspeed import comm as dist
from deepspeed import DeepSpeedConfig
from deepspeed.accelerator import get_accelerator

from tqdm import tqdm

mel_stft_loss = auraloss.freq.MelSTFTLoss(24_000, device="cuda")

def center_crop(x, len):
	start = (x.shape[-1] - len) // 2
	stop = start + len
	return x[..., start:stop]

def left_crop(x, len):
	return x[..., 0:len]

_logger = logging.getLogger(__name__)
deepspeed._initialized_dist = False

def load_engines():
	if not deepspeed._initialized_dist:
		deepspeed._initialized_dist = True
		deepspeed.init_distributed()

	models = get_models(cfg.models.get())
	engines = dict()

	for name in models:
		model = models[name]

		optimizer = None
		lr_scheduler = None

		Adam = ml.Adam
		AdamW = ml.AdamW

		if cfg.hyperparameters.optimizer.lower() == "adamw-torch":
			optimizer = AdamW(
				model.parameters(),
				lr=cfg.hyperparameters.learning_rate,
				betas=(0.9, 0.96),
				eps=1e-07,
				weight_decay=0.01,
			)

		if cfg.trainer.load_state_dict:
			load_dir = cfg.ckpt_dir / name / "fp32.pth"
			model.load_state_dict(torch.load(load_dir))

		ds_cfg=cfg.get_ds_cfg(model=model)
		config_class=DeepSpeedConfig(ds_cfg)
		engines[name] = trainer.Engine(
			model=model,
			config=ds_cfg,
			config_class=config_class,
			optimizer=optimizer,
			lr_scheduler=lr_scheduler,
		)

	return trainer.load_engines(engines, cfg)

def main():
	#dist.init_distributed(dist_backend=get_accelerator().communication_backend_name())
	if not deepspeed._initialized_dist:
		deepspeed._initialized_dist = True
		deepspeed.init_distributed()

	setup_logging(cfg.log_dir)

	train_dl, subtrain_dl, val_dl = create_train_val_dataloader()

	def train_feeder(engines, batch, name):
		stats = {}
		model = engines[name]
		if name.startswith("ar"):
			_ = model(
				text_list=batch["text"],
				proms_list=batch["proms"],
				resp_list=[r[..., 0] for r in batch["resps"]],
			)
		elif name.startswith("nar"):
			_ = model(
				text_list=batch["text"],
				proms_list=batch["proms"],
				resps_list=batch["resps"],
			)
		else:
			raise NotImplementedError(name)

		losses = model.gather_attribute("loss")

		loss = torch.stack([*losses.values()]).sum()

		stats = {}
		stats |= {k: v.item() for k, v in losses.items()}
		stats |= engines.gather_attribute("scalar")

		return loss, stats

	@torch.inference_mode()
	def run_eval(engines, eval_name, dl):
		engines_stats = {  
			'eval': eval_name
		}

		AR = None
		NAR = None

		names = []
		for name in engines:
			model = engines[name]
			names.append(name)
			if name[:2] == "ar":
				AR = model
			elif name[:3] == "nar":
				NAR = model

		stats = defaultdict(list)
		stats['loss'] = []
		
		for batch in tqdm(dl):
			batch: dict = to_device(batch, cfg.device)

			# if we're training both models, provide output for both
			if AR is not None and NAR is not None:
				name = "+".join(names)

				resp_list = AR(text_list=batch["text"], proms_list=batch["proms"], max_steps=cfg.evaluation.steps, sampling_temperature=cfg.evaluation.temperature)
				resps_list = [ r.unsqueeze(-1) for r in resp_list ]
				resps_list = NAR(text_list=batch["text"], proms_list=batch["proms"], resps_list=resps_list, sampling_temperature=cfg.evaluation.temperature)

				for speaker, path, ref, hyp, prom in zip(batch["spkr_name"], batch["path"], batch["resps"], resps_list, batch["proms"]):
					if len(hyp) == 0:
						continue

					filename = f'{speaker}_{path.parts[-1]}'

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

					min_length = min( ref_audio.shape[-1], hyp_audio.shape[-1] )
					ref_audio = ref_audio[..., 0:min_length]
					hyp_audio = hyp_audio[..., 0:min_length]
						
					stats['loss'].append(mel_stft_loss(hyp_audio, ref_audio).item())
			else:
				for name in engines:
					model = engines[name]

					if name.startswith("ar"):
						resp_list = model(
							text_list=batch["text"],
							proms_list=batch["proms"],
							max_steps=cfg.evaluation.steps,
							sampling_temperature=cfg.evaluation.temperature,
						)
						resps_list = [r.unsqueeze(-1) for r in resp_list]
					elif name.startswith("nar"):
						resps_list = model(
							text_list=batch["text"],
							proms_list=batch["proms"],
							resps_list=[r[..., 0].unsqueeze(-1) for r in batch["resps"]],
							sampling_temperature=cfg.evaluation.temperature,
						)
					else:
						raise NotImplementedError(name)

					losses = model.gather_attribute("loss")

					batch_stats = {}
					batch_stats |= {k: v.item() for k, v in losses.items()}
					batch_stats |= engines.gather_attribute("scalar")

					for k, v in batch_stats.items():
						stats[k].append(v)

					for speaker, path, ref, hyp, prom in zip(batch["spkr_name"], batch["path"], batch["resps"], resps_list, batch["proms"]):
						if len(hyp) == 0:
							continue

						filename = f'{speaker}_{path.parts[-1]}'

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
						
						stats['loss'].append(mel_stft_loss(hyp_audio, ref_audio).item())

		stats = {k: sum(v) / len(v) for k, v in stats.items()}
		engines_stats.update(flatten_dict({ name: stats }))

		iteration = engines.global_step
		engines_stats['it'] = iteration
		engines_stats['epoch'] = iteration * cfg.hyperparameters.gradient_accumulation_steps / len(train_dl)

		_logger.info(f"Validation Metrics: {json.dumps(engines_stats)}.")

	def eval_fn(engines):
		try:
			run_eval(engines, "subtrain", subtrain_dl)
			run_eval(engines, "val", val_dl)
		except Exception as e:
			print("Error occurred while performing eval:", str(e))
			print(traceback.format_exc())

		qnt.unload_model()
		do_gc()

	qnt.unload_model()

	trainer.train(
		engines_loader=load_engines,
		train_dl=train_dl,
		train_feeder=train_feeder,
		eval_fn=eval_fn,
	)


if __name__ == "__main__":
	main()