from .ar import AR
from .nar import NAR
from .ar_nar import AR_NAR

def get_model(cfg):
	if cfg.name == "ar":
		Model = AR
	elif cfg.name == "nar":
		Model = NAR
	elif cfg.name == "ar+nar":
		Model = AR_NAR
	else:
		raise f"invalid model name: {cfg.name}"
	name = cfg.name

	model = Model(
		n_tokens=cfg.tokens,
		d_model=cfg.dim,
		n_heads=cfg.heads,
		n_layers=cfg.layers,
		
		config = cfg,
	)
	model._cfg = cfg

	print(f"{name} ({next(model.parameters()).dtype}): {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")

	return model

def get_models(models):
	return { model.full_name: get_model(model) for model in models }
