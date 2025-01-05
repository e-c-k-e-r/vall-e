"""
# Helper script to grab all phonemes through parsed dataset metadata to find the "best" tokenizer dict
"""

import os
import json
import torch
import torchaudio

from tqdm.auto import tqdm
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from vall_e.config import cfg
from vall_e.utils.io import json_read
from vall_e.emb.g2p import coerce_to_hiragana

input_metadata = "training/metadata/"

output_file = Path("./training/tokenizer_pretraining_data.json")
tokenizer_data = []

def pad(num, zeroes):
	return str(num).zfill(zeroes+1)

def add( dir, type="training", audios=True, texts=True ):
	name = str(dir)
	name = name.replace(str(cfg.data_dir), "")
	speaker_name = name
	"""
	if "LibriTTS-R" in speaker_name:
		speaker_name = speaker_name.replace("LibriTTS-R", "LibriVox")
	"""

	metadata_path = cfg.metadata_dir / f'{speaker_name}.json'
	metadata = json_read( metadata_path, default={} )

	for k, entry in metadata.items():
		if "text" not in entry:
			continue

		language = entry.get('language','auto')
		text = entry['text']
		tokenizer_data.append( text )

if output_file.exists():
	tokenizer_data = json.loads(open(str(output_file), "r", encoding="utf-8").read())
else:
	# training
	for data_dir in tqdm(sorted(cfg.dataset.training), desc="Processing Training"):
		try:
			add( data_dir, type="training" )
		except Exception as e:
			pass

	# validation
	for data_dir in tqdm(sorted(cfg.dataset.validation), desc='Processing Validation'):
		try:
			add( data_dir, type="validation" )
		except Exception as e:
			pass
	"""
	for dataset_name in os.listdir(f'./{input_metadata}/'):
		if not os.path.isdir(f'./{input_metadata}/{dataset_name}/'):
			continue

		for speaker_id in tqdm(os.listdir(f'./{input_metadata}/{dataset_name}/'), desc="Processing speaker"):
			if not os.path.isdir(f'./{input_metadata}/{dataset_name}/{speaker_id}'):
				continue
					
			for id in os.listdir(f'./{input_metadata}/{dataset_name}/{speaker_id}/'):
				if ".json" not in id:
					continue

				metadata_path = Path(f'./{input_metadata}/{dataset_name}/{speaker_id}/{id}')
				metadata = json.loads(open(metadata_path, "r", encoding="utf-8").read())

				if "text" not in metadata:
					continue

				tokenizer_data.append( f'{"".join(metadata["text"])}' )

	open(output_file, 'w', encoding='utf-8').write(json.dumps(tokenizer_data))
	"""

unk_token = "<unk>"
spl_tokens = [unk_token, "<bos>", "</eos>", "<mask>", "<space>"]

trainer = BpeTrainer(special_tokens = spl_tokens, vocab_size = 32768, max_token_length=1, min_frequency=len(tokenizer_data))
tokenizer = Tokenizer(BPE(unk_token = unk_token))
tokenizer.pre_tokenizer = Whitespace() # takes 2 hours to process without this, we'll just manually add spaces as a token
tokenizer.post_processor = TemplateProcessing(
    single="<bos> $A <eos>",
    special_tokens=[("<bos>", 1), ("<eos>", 2)],
)

tokenizer.train_from_iterator(tokenizer_data, trainer=trainer)
tokenizer.save("./training/tokenizer_training_data.json")