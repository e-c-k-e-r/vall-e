# https://github.com/kyegomez/BitNet
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from bitnet.bit_transformer import Transformer as BitNetTransformerBlock, RMSNorm as BitNetRMSNorm

# re-enable logging because zetascale fucking sucks
import logging
logging.getLogger().setLevel(logging.DEBUG)

# override for wrapping checkpointing
def BitNetTransformerBlock_forward(self, x: Tensor, *args, **kwargs) -> Tensor:
	skip = x
	for attn, ffn in zip(self.layers, self.ffn_layers):
		if x.requires_grad and self.gradient_checkpointing:
			x, _ = checkpoint(attn, x, x, x, is_causal=True, *args, **kwargs, use_reentrant=False)
		else:
			x, _ = attn(x, x, x, is_causal=True, *args, **kwargs)
		x = x + skip
		x = ffn(x) + x
	return x

BitNetTransformerBlock.forward = BitNetTransformerBlock_forward

# override because bitnet's BitNetTransformer includes an embedding input / classifier output layers inside of it, which isn't favorable
class BitNetTransformer(nn.Module):
	def __init__(
		self,
		dim: int,
		depth: int,
		num_tokens: int,
		heads=8,
		ff_mult=4,
		gradient_checkpointing = True
	):
		super().__init__()

		self.transformer = BitNetTransformerBlock( dim=dim, depth=depth, heads=heads, ff_mult=ff_mult )
		self.norm = BitNetRMSNorm(dim)
		self.transformer.gradient_checkpointing = gradient_checkpointing

	def forward(self, x):
		x = self.transformer(x)
		return self.norm( x )

"""
from bitnet import BitNetTransformer
def NoEmbedding_BitNetTransformer_Forward(self, x):
	x = self.transformer(x)
	return self.to_logits[0](x)

BitNetTransformer.forward = NoEmbedding_BitNetTransformer_Forward 
"""