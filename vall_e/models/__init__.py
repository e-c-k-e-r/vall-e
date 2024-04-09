from .ar import AR
from .nar import NAR
from .ar_nar import AR_NAR

def get_model(cfg, training=True):
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
		n_experts=cfg.experts,
		
		l_padding = cfg.input_alignment,
		
		training = training,
		config = cfg,
	)
	model._cfg = cfg

	print(f"{name} ({next(model.parameters()).dtype}): {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")

	return model

def get_models(models, training=True):
	return { model.full_name: get_model(model, training=training) for model in models }
