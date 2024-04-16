from .ar_nar import AR_NAR

def get_model(cfg, training=True):
	name = cfg.name

	model = AR_NAR(
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
