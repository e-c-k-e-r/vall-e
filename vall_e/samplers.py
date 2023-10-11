import math
import torch
import torch.nn.functional as F
import numpy as np

from torch import Tensor, einsum, nn

# Simple filter to modify a token's probability if it shows up in the past
# `one_time` will only apply the penalty once
# `decay` is a factor that will exponentially apply to how far away it is
def reptition_penalize( logits, previous, factor=1.0, decay=0.0, one_time=True ):
	if factor == 1.0 or previous is None:
		return logits

	unique = set()
	priors = reversed(previous.tolist())
	for distance, token in enumerate(priors):
		# skip if we're only applying the decay once
		if one_time and token in unique:
			continue

		distance += 1
		logits[:, token] /= factor * (distance ** decay)
		
		# add to set if we care about it
		if one_time:
			unique.add(token)

	return logits

# Simple "filter" that modifies the logit for the stop token, based on the sequence length
# `length` is the length of the sequence currently
# `factor` is the power the length is raised to, so values > 0 will yield longer sequences, values < 0 will yield shorter sequences
# `token` is the stop token.
def length_penalize( logits, length, factor=0.0, token=-1 ):
	if factor == 0.0:
		return logits

	logits[:, token] /= (length ** factor)
	return logits

# Credit to https://github.com/microsoft/unilm/blob/master/xtune/src/transformers/modeling_utils.py#L1145 / https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
def top_k_top_p_filtering( logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens=1 ):
	"""Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
	Args:
		logits: logits distribution shape (batch size, vocabulary size)
		if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
		if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
			Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
		Make sure we keep at least min_tokens per batch example in the output
	"""
	if top_k > 0:
		top_k = min(max(top_k, min_tokens), logits.size(-1))  # Safety check
		# Remove all tokens with a probability less than the last token of the top-k
		indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
		logits[indices_to_remove] = filter_value

	if top_p < 1.0:
		sorted_logits, sorted_indices = torch.sort(logits, descending=True)
		cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

		# Remove tokens with cumulative probability above the threshold (token with 0 are kept)
		sorted_indices_to_remove = cumulative_probs > top_p
		if min_tokens > 1:
			# Keep at least min_tokens (set to min_tokens-1 because we add the first one below)
			sorted_indices_to_remove[..., :min_tokens] = 0
		# Shift the indices to the right to keep also the first token above the threshold
		sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
		sorted_indices_to_remove[..., 0] = 0

		# scatter sorted tensors to original indexing
		indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
		logits[indices_to_remove] = filter_value

	return logits

# credit to https://github.com/LostRuins/koboldcpp/pull/464 // https://github.com/kalomaze/koboldcpp/tree/dynamic-temp
def dynamic_temperature( logits, temperature=1.0, min_temperature = 0.0, k = 10, sigmoidCenterPoint = 0.5 ):
	# loop over logits[:], as the NAR will have logits.shape[0] > 1
	for i in range(logits.shape[0]):
		sum_exp = 0.0
		maximum = torch.max( logits[i] )
		for logit in logits[i]:
			sum_exp += math.exp( logit - maximum )

		prob_max_token_before_temp = 1.0 / sum_exp
		dynamic_temperature = temperature - (temperature - min_temperature) / (1 + math.exp(-k * (prob_max_token_before_temp - sigmoidCenterPoint)))

		logits[i] /= dynamic_temperature

	return logits



# picks the top K tokens amongst a batch of logits
# logits: [Tensor] list of logits
# candidates: [(batch, token)] list, where batch indicates the index of the logits the given token is from
def top_k_logits_list( logits_list, k ):
	# ( batch, tokens ) => ( batch x tokens )
	logits = torch.cat( logits_list )
	candidates = list(torch.topk(logits.flatten(), k).indices.tolist()) # perform top-k across all logits
	for i, index in enumerate(candidates):
		t = []
		N = np.prod(logits.size())
		for n in logits.size():
			N //= n
			t.append(index // N)
			index %= N
		candidates[i] = tuple(t)
	return candidates


# Credit to: https://github.com/basusourya/mirostat/
# performs mirostat-based sampling
# logits: Tensor of logit probabilities
# state: the mirostat state
def mirostat_sample( logits, state = None ):
	def compute_k(prob, n, tau):
		num = 0
		den = 0
		for i in range(100):
			b = prob[i]/prob[i+1]
			t = (i+2)/(i+1)
			num += math.log(b)*math.log(t)
			den += math.log(t)**2
				
		s = num/den
		eps = s-1
		k = ((eps*(2**(tau)))/(1-n**(-eps)))**(1/s)
		k = round(k)
		return k

	if "max_surprise" not in state:
		state["max_surprise"] = state["tau"] * 2

	if "error_surprise" not in state:
		state["error_surprise"] = 0

	if "running_total_surprise" not in state:
		state["running_total_surprise"] = 0
	
	sorted_logits, sorted_indices = torch.sort( logits[-1, :], descending=True )
	prob_original = torch.softmax( sorted_logits, dim=-1 ).tolist()

	k = compute_k(prob_original, state["n"], state["max_surprise"]) + 1

	sorted_logits = sorted_logits[0:k]
	sorted_indices = sorted_indices[0:k]
	prob_topk = torch.softmax(sorted_logits, dim = 0)
	prev_i = torch.multinomial(prob_topk, num_samples=1, replacement=True)
	
	state["index_surprise"] = math.log2(1/prob_original[prev_i])
	state["running_total_surprise"] += state["index_surprise"]
	state["error_surprise"] = state["index_surprise"] - state["tau"]
	state["max_surprise"] -= state["eta"] * state["error_surprise"]
	state["token"] = sorted_indices[prev_i]

	return state