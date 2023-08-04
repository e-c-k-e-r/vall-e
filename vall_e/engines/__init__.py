from ..config import cfg

if cfg.trainer.backend == "deepspeed":
	from .deepspeed import Engine
elif cfg.trainer.backend == "local":
	from .base import Engine

from .base import Engines, TrainFeeder, default_feeder