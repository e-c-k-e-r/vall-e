from .ar import AR
from .nar import NAR

def get_model(cfg):
	if cfg.name == "ar":
		Model = AR
	elif cfg.name == "nar":
		Model = NAR
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

	print(f"{name} parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

	return model

def get_models(models):
	return { model.full_name: get_model(model) for model in models }
