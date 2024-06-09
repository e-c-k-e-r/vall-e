
def get_model(config, training=True):
	name = config.name

	if not config.experimental:
		from .ar_nar import AR_NAR
		model = AR_NAR(
			n_text_tokens=config.text_tokens,
			n_audio_tokens=config.audio_tokens,
			d_model=config.dim,
			n_heads=config.heads,
			n_layers=config.layers,
			n_experts=config.experts,
			
			p_dropout=config.dropout,
			
			l_padding = config.input_alignment,
			
			training = training,
			config = config,
		)
	elif "len" in config.capabilities:
		from .nar import NAR
		model = NAR(
			n_text_tokens=config.text_tokens,
			n_audio_tokens=config.audio_tokens,
			d_model=config.dim,
			n_heads=config.heads,
			n_layers=config.layers,
			n_experts=config.experts,
			
			p_dropout=config.dropout,
			
			l_padding = config.input_alignment,
			
			training = training,
			config = config,
		)
	else:
		from .experimental import Model as Experimental
		model = Experimental(
			n_text_tokens=config.text_tokens,
			n_audio_tokens=config.audio_tokens,

			d_model=config.dim,
			n_layers=config.layers,
			n_heads=config.heads,
			p_dropout=config.dropout,

			config = config,
		)

	print(f"{name} ({next(model.parameters()).dtype}): {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")

	return model

def get_models(models, training=True):
	return { model.full_name: get_model(model, training=training) for model in models }
