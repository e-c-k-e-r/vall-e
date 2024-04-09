# https://github.com/microsoft/torchscale

from torchscale.architecture.config import RetNetConfig
from torchscale.architecture.retnet import RetNetDecoder
# from retnet import RetNet

# override MultiScaleRetention's forward because training with te throws an error
from torchscale.component.multiscale_retention import MultiScaleRetention, theta_shift

def MultiScaleRetention_forward(
		self,
		x,
		rel_pos,
		chunkwise_recurrent=False,
		incremental_state=None
	):
		bsz, tgt_len, _ = x.size()
		(sin, cos), inner_mask = rel_pos

		q = self.q_proj(x)
		k = self.k_proj(x) * self.scaling
		v = self.v_proj(x)
		g = self.g_proj(x)

		q = q.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
		k = k.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)

		qr = theta_shift(q, sin, cos)
		kr = theta_shift(k, sin, cos)

		if incremental_state is not None:
			output = self.recurrent_forward(qr, kr, v, inner_mask, incremental_state)
		elif chunkwise_recurrent:
			output = self.chunk_recurrent_forward(qr, kr, v, inner_mask)
		else:
			output = self.parallel_forward(qr, kr, v, inner_mask)
		
		output = self.group_norm(output).reshape(bsz, tgt_len, self.head_dim * self.num_heads)

		output = self.gate_fn(g) * output

		output = self.out_proj(output)

		return output

MultiScaleRetention.forward = MultiScaleRetention_forward