# Models from the Hugging Face hub, cache first: a cached copy is used without touching the network (a hiccup at
# huggingface.co once failed a long evaluation), and a first run downloads it.


def snapshot(repo, allow=("*.json", "*.txt", "*.safetensors")):
	"""A hub model repository's local directory (just the files `allow` matches: configs, tokenizers, safetensors)."""
	from huggingface_hub import snapshot_download
	try:
		return snapshot_download(repo, allow_patterns=list(allow), local_files_only=True)
	except Exception:
		return snapshot_download(repo, allow_patterns=list(allow))
