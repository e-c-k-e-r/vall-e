from fairseq.models import FairseqIncrementalDecoder
from fairseq.incremental_decoding_utils import with_incremental_state

from torchscale.architecture.config import RetNetConfig
from torchscale.architecture.retnet import RetNetDecoder as Decoder

@with_incremental_state
class RetNetDecoder(Decoder):
	def forward(self, src_tokens, **kwargs):
		return super().forward(src_tokens, **kwargs)

	def max_positions(self):
		return self.args.max_token_positions

	def reorder_incremental_state( self, incremental_state, new_order ):
		for module in incremental_state:
			for key in incremental_state[module]:
				result = incremental_state[module][key].index_select(0, new_order)
				incremental_state[module][key] = result