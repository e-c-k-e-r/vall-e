# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py

import torch

from transformers import MixtralModel, MixtralConfig
from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func, MixtralSparseMoeBlock

# This is required because batch sizes > 1 throws errors
def Fixed_MixtralSparseMoeBlock_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
	""" """
	batch_size, sequence_length, hidden_dim = hidden_states.shape
	hidden_states = hidden_states.reshape(-1, hidden_dim) # was view()
	# router_logits: (batch * sequence_length, n_experts)
	router_logits = self.gate(hidden_states)

	routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
	routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
	routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
	# we cast back to the input dtype
	routing_weights = routing_weights.to(hidden_states.dtype)

	final_hidden_states = torch.zeros(
		(batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
	)

	expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

	for expert_idx in range(self.num_experts):
		expert_layer = self.experts[expert_idx]
		idx, top_x = torch.where(expert_mask[expert_idx])

		if top_x.shape[0] == 0:
			continue
		top_x_list = top_x.tolist()
		idx_list = idx.tolist()

		current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
		current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

		final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
	final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
	return final_hidden_states, router_logits

Original_MixtralSparseMoeBlock_forward = MixtralSparseMoeBlock.forward
MixtralSparseMoeBlock.forward = Fixed_MixtralSparseMoeBlock_forward