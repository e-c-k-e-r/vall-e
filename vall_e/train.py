# todo: clean this mess up

from .config import cfg
from .data import create_train_val_dataloader
from .emb import qnt

from .utils import setup_logging, to_device, trainer, flatten_dict, do_gc

import auraloss
import json
import logging
import random
import torch
import torch.nn.functional as F
import traceback

from collections import defaultdict

from tqdm import tqdm

mel_stft_loss = auraloss.freq.MelSTFTLoss(24_000, device="cpu")
_logger = logging.getLogger(__name__)

def train_feeder(engine, batch):
	with torch.autocast("cuda", dtype=cfg.trainer.dtype, enabled=cfg.trainer.amp):
		engine(
			text_list=batch["text"],
			proms_list=[prom[:, :engine._cfg.prom_levels] for prom in batch["proms"]], # reduce the input prompt to the target prom level
			resps_list=batch["resps"],
			lang_list=batch["lang"],
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
	AR = None
	NAR = None
	AR_NAR = None

	names = []
	for name, engine in engines.items():
		if name[:6] == "ar+nar":
			AR_NAR = engine
		elif name[:2] == "ar":
			AR = engine
		elif name[:3] == "nar":
			NAR = engine
		else:
			continue
		names.append(name)

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
		batch: dict = to_device(next(iter(dl)), cfg.device)
		processed += len(batch["text"])

		# if we're training both models, provide output for both
		if AR is not None and NAR is not None:
			name = "+".join(names)

			resps_list = AR(text_list=batch["text"], proms_list=batch["proms"], max_steps=cfg.evaluation.steps, sampling_temperature=cfg.evaluation.ar_temperature)
			resps_list = [ r.unsqueeze(-1) for r in resps_list ]
			resps_list = NAR(text_list=batch["text"], proms_list=batch["proms"], resps_list=resps_list, sampling_temperature=cfg.evaluation.nar_temperature)

			process( name, batch, resps_list )
		else:
			for name in engines:
				model = engines[name]

				if name.startswith("ar+nar"):
					resps_list = AR_NAR(text_list=batch["text"], proms_list=batch["proms"], max_steps=cfg.evaluation.steps, sampling_temperature=cfg.evaluation.ar_temperature)
					resps_list = [ r.unsqueeze(-1) for r in resps_list ]
					resps_list = AR_NAR(text_list=batch["text"], proms_list=batch["proms"], resps_list=resps_list, sampling_temperature=cfg.evaluation.nar_temperature)
				elif name.startswith("ar"):
					resps_list = model(
						text_list=batch["text"],
						proms_list=batch["proms"],
						max_steps=cfg.evaluation.steps,
						sampling_temperature=cfg.evaluation.ar_temperature,
					)
					resps_list = [r.unsqueeze(-1) for r in resps_list]
				elif name.startswith("nar"):
					resps_list = model(
						text_list=batch["text"],
						proms_list=batch["proms"],
						resps_list=[r[..., 0].unsqueeze(-1) for r in batch["resps"]],
						sampling_temperature=cfg.evaluation.nar_temperature,
					)
				else:
					raise NotImplementedError(name)

				process( name, batch, resps_list )


	stats = {k: sum(v) / len(v) for k, v in stats.items()}
	engines_stats = {
		f'{name}.{eval_name}': stats,
		"it": engines.global_step,
	}
	#engines_stats['epoch'] = iteration * cfg.hyperparameters.gradient_accumulation_steps / len(dl)

	_logger.info(f"Validation Metrics: {json.dumps(engines_stats)}.")


def main():
	setup_logging(cfg.log_dir)

	train_dl, subtrain_dl, val_dl = create_train_val_dataloader()
	
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
		train_dl=train_dl,
		train_feeder=train_feeder,
		eval_fn=eval_fn,
	)

if __name__ == "__main__":
	main()
