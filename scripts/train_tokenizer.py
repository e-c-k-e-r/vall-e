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

input_metadata = "training/data"

output_file = Path("./training/tokenizer_training_data.json")
tokenizer_data = []

def pad(num, zeroes):
	return str(num).zfill(zeroes+1)

if output_file.exists():
	tokenizer_data = json.loads(open(str(output_file), "r", encoding="utf-8").read())
else:
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

				if "phonemes" not in metadata:
					continue

				tokenizer_data.append( f'{"".join(metadata["phonemes"])}' )

	open(output_file, 'w', encoding='utf-8').write(json.dumps(tokenizer_data))

unk_token = "<unk>"
spl_tokens = [unk_token, "<bos>", "</eos>", "<mask>", "<space>"]

trainer = BpeTrainer(special_tokens = spl_tokens, vocab_size = 256)
tokenizer = Tokenizer(BPE(unk_token = unk_token))
tokenizer.pre_tokenizer = Whitespace() # takes 2 hours to process without this, we'll just manually add spaces as a token
tokenizer.post_processor = TemplateProcessing(
    single="<bos> $A <eos>",
    special_tokens=[("<bos>", 1), ("<eos>", 2)],
)

tokenizer.train_from_iterator(tokenizer_data, trainer=trainer)
tokenizer.save("./training/tokenizer_training_data.json")