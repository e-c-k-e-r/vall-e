
def get_model(cfg, training=True):
	name = cfg.name

	if not cfg.experimental:
		from .ar_nar import AR_NAR
		model = AR_NAR(
			n_text_tokens=cfg.text_tokens,
			n_audio_tokens=cfg.audio_tokens,
			d_model=cfg.dim,
			n_heads=cfg.heads,
			n_layers=cfg.layers,
			n_experts=cfg.experts,
			
			p_dropout=cfg.dropout,
			
			l_padding = cfg.input_alignment,
			
			training = training,
			config = cfg,
		)
		model._cfg = cfg
	else:
		from .experimental import Model as Experimental
		model = Experimental(
			n_text_tokens=cfg.text_tokens,
			n_audio_tokens=cfg.audio_tokens,
			
			d_model=cfg.dim,
			n_layers=cfg.layers,
			n_heads=cfg.heads,
			p_dropout=cfg.dropout,

			config = cfg,
		)

	print(f"{name} ({next(model.parameters()).dtype}): {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")

	return model

def get_models(models, training=True):
	return { model.full_name: get_model(model, training=training) for model in models }
