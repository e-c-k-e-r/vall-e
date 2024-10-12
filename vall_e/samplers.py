import math
import torch
import torch.nn.functional as F
import numpy as np

from torch import Tensor, einsum, nn

from dataclasses import asdict, dataclass, field

# Simple filter to modify a token's probability if it shows up in the past
# `one_time` will only apply the penalty once
# `decay` is a factor that will exponentially apply to how far away it is
def reptition_penalize( logits, previous, factor=1.0, decay=0.0, one_time=True ):
	if factor == 1.0 or previous is None:
		return logits

	unique = set()
	priors = reversed(previous)
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

# Simple way to ban tokens
def ban_tokens( logits, tokens ):
	for token in tokens:
		# token not in logits
		if logits.shape[-1] >= token:
			continue
		logits[:, token] = -float("inf")
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

# Credits to: https://github.com/oobabooga/text-generation-webui/pull/5677
# performs DRY sampling
# * (honestly it looks close to rep pen anyways but what do I know)
# `logits` are the scores used to sample against
# `previous` are the prior tokens to penalize with
# `factor` is the scalar multiplier
# `base` is the base number to raise to the (length - allowed_length)th power
# `allowed_length` limits the range to apply DRY to
def dry_sampling( logits, previous=None, factor=0.0, base=1.75, allowed_length=2 ):
	if factor == 0.0 or previous is None:
		return logits

	lengths = {}
	for i, token in enumerate( previous ):
		length = 1
		while length < max(allowed_length, 50):
			j = i - length
			
			# Start of input reached.
			if j < 0:
				break

			# Start of match reached.
			if previous[j] != previous[-length-1]:
				break

			length += 1

		lengths[token] = max(length, lengths[token]) if token in lengths else length

	for token, length in lengths.items():
		if length < allowed_length:
			break
		logits[:, token] -= factor * base ** (length - allowed_length)

	return logits


LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E

# Grabbed from https://github.com/xjdr-alt/entropix/blob/main/entropix/sampler.py
# Right now I only care about quantifying these two, I'll figure out how to best apply this to the model
def calculate_entropix_metrics( logits, attention_scores=None, dim=-1 ):
	"""Calculate the entropy and varentropy of the probability distribution using logsoftmax."""
	log_probs = torch.nn.functional.log_softmax(logits, dim=dim)
	probs = torch.exp(log_probs)
	entropy = -torch.sum(probs * log_probs, dim=dim) / LN_2  # Convert to base-2
	varentropy = torch.sum(probs * (log_probs / LN_2 + entropy[..., None])**2, dim=dim)

	if attention_scores is None:
		return {
			"logits_entropy": torch.mean(entropy).item(),
			"logits_varentropy": torch.mean(varentropy).item(),
		}

	attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
	attn_entropy = -torch.sum(attention_probs * torch.log2(torch.clip(attention_probs, 1e-10, 1.0)), dim=-1)
	attn_varentropy = torch.var(attn_entropy, dim=1)

	mean_attention = torch.mean(attention_probs, dim=1)
	agreement = torch.mean(torch.abs(attention_probs - mean_attention[:, None, :]), dim=(1, 2))

	interaction_strength = torch.mean(torch.abs(attention_scores), dim=(1, 2, 3))
	return {
		"logits_entropy": torch.mean(entropy),
		"logits_varentropy": torch.mean(varentropy),
		"attn_entropy": torch.mean(attn_entropy),
		"attn_varentropy": torch.mean(attn_varentropy),
		"agreement": torch.mean(agreement),
		"interaction_strength": torch.mean(torch.abs(attention_scores), dim=(1, 2, 3)),
	}

# to-do: play around with these values
@dataclass()
class EntropixSamplerConfig:
    temp: float = 0.999
    top_p: float = 0.90
    top_k: int = 32
    min_p: float = 0.01 # was 0.03  # Turn this down to 0.01 to reduce the shoggoth

    low_ent_thresh: float = 0.1
    low_vent_thresh: float = 0.1
    med_ent_thresh: float = 3.0
    high_ent_thresh: float = 5.0
    high_vent_thresh: float = 5.0

    # TODO this is a bit of a nasty mess, but also makes all the hyperparameters visible
    helv_attn_ent_offset: float = 1.3
    helv_attn_ent_coef: float = 0.2

    lehv_interaction_strength_offset: float = 1.2
    lehv_interaction_strength_coef: float = 0.3

    hehv_attn_ent_coef: float = 0.2
    hehv_attn_vent_offset: float = 2.0
    hehv_attn_vent_coef: float = 0.5

    # TODO not convinced this should
    n_adaptive_samples: int = 5

    # Adaptive sampling parameters
    ada_temp_logits: float = 0.3
    ada_temp_attn: float = 0.2
    ada_temp_agree: float = 0.2
    ada_top_p: float = 0.1
    ada_top_k_int: float = 0.3
    ada_top_k_agree: float = 0.2
    ada_min_p: float = 0.5
    ada_score_logits_ent: float = 0.1
    ada_score_attn_ent: float = 0.2
    ada_score_logits_vent: float = 0.3
    ada_score_attn_vent: float = 0.4
    ada_score_agree: float = 0.5
    ada_score_int: float = 0.6

    # extra stuff
    top_k_min: int = 32
    top_k_max: int = 128