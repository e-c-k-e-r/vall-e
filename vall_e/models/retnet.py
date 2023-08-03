"""
# https://github.com/facebookresearch/fairseq/blob/main/fairseq/incremental_decoding_utils.py
# Copied directly because even having fairseq installed WILL break logging, why are corposhitters like this
"""

import uuid
from typing import Dict, Optional

from torch import Tensor

class FairseqIncrementalState(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_incremental_state()

    def init_incremental_state(self):
        self._incremental_state_id = str(uuid.uuid4())

    def _get_full_incremental_state_key(self, key: str) -> str:
        return "{}.{}".format(self._incremental_state_id, key)

    def get_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
    ) -> Optional[Dict[str, Optional[Tensor]]]:
        """Helper for getting incremental state for an nn.Module."""
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
        value: Dict[str, Optional[Tensor]],
    ) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
        """Helper for setting incremental state for an nn.Module."""
        if incremental_state is not None:
            full_key = self._get_full_incremental_state_key(key)
            incremental_state[full_key] = value
        return incremental_state


def with_incremental_state(cls):
    cls.__bases__ = (FairseqIncrementalState,) + tuple(
        b for b in cls.__bases__ if b != FairseqIncrementalState
    )
    return cls


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