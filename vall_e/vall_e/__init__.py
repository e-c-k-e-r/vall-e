from .ar import AR
from .nar import NAR

def get_model(model):
	if model.name == "ar":
		Model = AR
	elif model.name == "nar":
		Model = NAR
	else:
		raise f"invalid model name: {model.name}"
	name = model.name
	model = Model(
		n_tokens=model.tokens,
		d_model=model.dim,
		n_heads=model.heads,
		n_layers=model.layers,
	)

	print(f"{name} parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

	return model

def get_models(models):
	return { model.full_name: get_model(model) for model in models }
