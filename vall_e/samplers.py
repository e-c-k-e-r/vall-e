import math
import torch
import torch.nn.functional as F
import numpy as np
import time

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

# Performs min_p filtering
# From https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/generation/logits_process.py#L537
def min_p_filtering( logits, min_p=0.0, min_tokens_to_keep=32 ):
	if min_p <= 0.0:
		return logits

	# Convert logits to probabilities
	probs = torch.softmax(logits, dim=-1)
	# Get the probability of the top token for each sequence in the batch
	top_probs, _ = probs.max(dim=-1, keepdim=True)
	# Calculate the actual min_p threshold by scaling min_p with the top token's probability
	scaled_min_p = min_p * top_probs

	sorted_indices = torch.argsort(logits, descending=True, dim=-1)
	sorted_indices_to_remove = torch.gather(probs < scaled_min_p, dim=-1, index=sorted_indices)
	sorted_indices_to_remove[..., :min_tokens_to_keep] = False

	indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
	return logits.masked_fill(indices_to_remove, -float("inf"))

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
def calculate_entropix_metrics( logits, attentions=None, dim=-1, use_stats=False ):
	"""Calculate the entropy and varentropy of the probability distribution using logsoftmax."""
	log_probs = F.log_softmax(logits, dim=dim)
	probs = torch.exp(log_probs)
	entropy = -torch.sum(probs * log_probs, dim=dim) / LN_2  # Convert to base-2
	varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1))**2, dim=dim)

	if attentions is None:
		return {
			"logits_entropy": torch.mean(entropy).item(),
			"logits_varentropy": torch.mean(varentropy).item(),
		}

	last_attention_scores = attentions[-1].unsqueeze(0) # ( bsz, heads, seq_len, seq_len )
	attention_probs = F.softmax(last_attention_scores, dim=-1)
	if use_stats:
		attn_stats = AttnStats.new( 1, attentions.shape[0], attentions.shape[1], logits.device )
		for idx, attn in enumerate( attentions ):
			attn_stats.update( attn.unsqueeze(0)[:, :, -1, :], idx ) # (bsz, heads, last_token, seq_len)
		attn_entropy = attn_stats.entropy
		attn_varentropy = attn_stats.varentropy
	else:
		attn_entropy = -torch.sum(attention_probs * torch.log2(torch.clamp(attention_probs, 1e-10, 1.0)), dim=-1)
		attn_varentropy = torch.var(attn_entropy, dim=1)

	# Add a small epsilon to avoid NaN when all values are the same
	attn_varentropy = torch.where(torch.isnan(attn_varentropy), torch.zeros_like(attn_varentropy), attn_varentropy)
	mean_attention = torch.mean(attention_probs, dim=1)
	agreement = torch.mean(torch.abs(attention_probs - mean_attention.unsqueeze(1)), dim=(1, 2))

	interaction_strength = torch.mean(torch.abs(last_attention_scores), dim=(1, 2, 3))

	return {
		"logits_entropy": torch.mean(entropy).item(),
		"logits_varentropy": torch.mean(varentropy).item(),
		"attn_entropy": torch.mean(attn_entropy).item(),
		"attn_varentropy": torch.mean(attn_varentropy).item(),
		"agreement": torch.mean(agreement).item(),
		"interaction_strength": interaction_strength.item(), # torch.mean(interaction_strength).item(),
		"action": -1
	}

from typing import NamedTuple
class AttnStats(NamedTuple):
	entropy: torch.Tensor  # (bsz, n_layers, num_heads)
	varentropy: torch.Tensor  # (bsz, n_layers, num_heads)
	n_layers: int
	n_heads: int

	@classmethod
	def new(cls, bsz: int, n_layers: int, n_heads: int, device = "cuda") -> 'AttnStats':
		return cls(
			entropy=torch.zeros((bsz, n_layers, n_heads), dtype=torch.float32, device=device),
			varentropy=torch.zeros((bsz, n_layers, n_heads), dtype=torch.float32, device=device),
			n_layers=n_layers,
			n_heads=n_heads
		)

	@property
	def avg_entropy(self):
		return self.entropy.sum(dim=-1, keepdim=False)  # Average across heads
	
	@property
	def avg_varentropy(self):
		return self.varentropy.sum(dim=-1, keepdim=False)  # Average across heads

	@property
	def std_error(self):
		return torch.sqrt(torch.mean(self.varentropy)) / (self.n_heads * self.n_layers)

	def update(self, scores: torch.Tensor, layer_idx: int):
		# scores shape: (bsz, n_heads, seqlen, n_words)
		probs = torch.nn.functional.softmax(scores, dim=-1)
		new_entropy = -torch.sum(torch.where(probs > 0, probs * torch.log(probs), torch.tensor(0.0)), dim=-1)
		new_varentropy = torch.sum(probs * (torch.log(probs) + new_entropy.unsqueeze(-1))**2, dim=-1)

		# Update entropy and varentropy tensors
		self.entropy[:, layer_idx, :] = new_entropy
		self.varentropy[:, layer_idx, :] = new_varentropy

		return self

# to-do: play around with these values
@dataclass()
class EntropixSamplerConfig:
	temp: float = 0.666
	top_p: float = 0.90
	top_k: int = 27
	min_p: float = 0.01 # was 0.03  # Turn this down to 0.01 to reduce the shoggoth

	low_ent_thresh: float = 0.1 # 3.0
	low_vent_thresh: float = 0.1 # 3.0
	med_ent_thresh: float = 3.0 # 6.0
	high_ent_thresh: float = 5.0 # 9.0
	high_vent_thresh: float = 5.0 # 9.0

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
	temperature_max: float = 1.25
	temperature_min: float = 0.5
	top_k_min: int = 1
	top_k_max: int = 1024
	top_p_min: int = 0.1
	top_p_max: int = 1.0
	min_p_min: int = 0.01
	min_p_max: int = 0.5

Exponential = torch.distributions.exponential.Exponential(1.0)

# Doing as close to the original sampling method just to reduce variance
def _sample_entropix(
	logits,
	temperature=1.0,
	top_k=0,
	top_p=1.0,
	min_p=0.0,
	cfg=EntropixSamplerConfig(),
):
	def clamp(n, lo, hi):
		return max(lo, min(n, hi))

	if top_k == 0:
		top_k = logits.shape[-1]

	logit = logits[-1, :]

	temperature = clamp( float(temperature), cfg.temperature_min, cfg.temperature_max )
	top_p = clamp( float(top_p), cfg.top_p_min, cfg.top_p_max )
	top_k = clamp( int(top_k), cfg.top_k_min, cfg.top_k_max )
	min_p = clamp( float(min_p), cfg.min_p_min, cfg.min_p_max )

	probs = F.softmax(logit / temperature, dim=-1)

	# Apply min_p sampling
	if min_p > 0.0:
		p_max = float(torch.max(probs, dim=-1, keepdim=True).values)
		indices_to_remove = probs < (min_p * p_max)
		logit = torch.where(indices_to_remove, torch.full_like(logit, float('-inf')), logit)

	# Apply top-k sampling
	top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.shape[-1]))
	probs_sort = torch.flip(top_k_probs, dims=[-1])
	probs_idx = torch.flip(top_k_indices, dims=[-1])
	probs_sum = torch.cumsum(probs_sort, dim=-1)
	# Apply top-p sampling
	mask = torch.where(probs_sum - probs_sort > top_p, torch.tensor(1.0, device=logit.device), torch.tensor(0.0, device=logit.device))
	probs_sort = probs_sort * (1 - mask)
	probs_sort = probs_sort / torch.sum(probs_sort, dim=-1, keepdim=True)

	q = Exponential.sample(probs_sort.shape)
	"""
	# q = torch.rand(probs_sort.shape, generator=generator, device=probs_sort.device)
	"""
	next_token = torch.argmax(probs_sort / q, dim=-1, keepdim=True)
	next_token_g = torch.take_along_dim(probs_idx, next_token, dim=-1)
	return next_token_g

def sample_entropix(
	logits,
	attentions,
	temperature=1.0,
	top_k=27,
	top_p=1.0,
	min_p=0.0,
	cfg=EntropixSamplerConfig(),
):
	"""
	temperature = cfg.temp
	top_k = cfg.top_k
	top_p = cfg.top_p
	"""

	# logits: ( seq_len, vocab )
	# attentions: ( layer, heads, seq_len, seq_len )
	metrics = calculate_entropix_metrics( logits[-1:, :], attentions[:, :, -1:, :] )

	ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
	attn_ent, attn_vent = metrics["attn_entropy"], metrics["attn_varentropy"]
	agreement = metrics["agreement"]
	interaction_strength = metrics["interaction_strength"]

	# Low Entropy, Low Varentropy: "flowing with unspoken intent"
	if ent < cfg.low_ent_thresh and vent < cfg.low_vent_thresh:
		metrics["action"] = 0
		res = logits[-1, :].argmax(dim=1)
	# High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
	elif ent > cfg.high_ent_thresh and vent < cfg.low_vent_thresh:
		metrics["action"] = 1
		# sample with slightly higher temperature
		temperature *= cfg.helv_attn_ent_offset + cfg.helv_attn_ent_coef * attn_ent  # Increase temperature based on attention entropy
		res = _sample_entropix( logits, temperature, top_k, top_p, min_p, cfg=cfg )
	# Low Entropy, High Varentropy: "exploring forks in the path"
	elif ent < cfg.high_ent_thresh and vent > cfg.high_vent_thresh:
		metrics["action"] = 2
		temperature *= cfg.lehv_interaction_strength_offset + cfg.lehv_interaction_strength_coef * interaction_strength  # Increase temperature based on interaction strength
		top_k = max(5, int(top_k * (1 + 0.5 * (1 - agreement))))  # Increase top_k when agreement is low
		res = _sample_entropix( logits, temperature, top_k, top_p, min_p, cfg=cfg )
	# High Entropy, High Varentropy: "resampling in the mist"
	elif ent > cfg.med_ent_thresh and vent > cfg.high_vent_thresh:
		metrics["action"] = 3
		# Use high temperature and adjusted top_p based on attention metrics
		temperature *= cfg.hehv_attn_vent_offset + cfg.hehv_attn_vent_coef * attn_vent  # Increase temperature based on attention varentropy
		top_p = max(0.5, top_p - cfg.hehv_attn_ent_coef * attn_ent)  # Decrease top_p when attention entropy is high
		res = _sample_entropix( logits, temperature, top_k, top_p, min_p, cfg=cfg )
	# Middle ground: use adaptive sampling
	else:
		metrics["action"] = 4

		log_softmax = F.log_softmax(logits, dim=-1)
		logits_uncertainty = ent + vent
		attn_uncertainty = attn_ent + attn_vent

		temperature *= 1 + cfg.ada_temp_logits * logits_uncertainty + cfg.ada_temp_attn * attn_uncertainty - cfg.ada_temp_agree * agreement
		top_p = top_p * (1 + cfg.ada_top_p * attn_vent)
		top_k = round(float(top_k * (1 + cfg.ada_top_k_int * interaction_strength - cfg.ada_top_k_agree * agreement)))
		min_p = cfg.min_p * (1 - cfg.ada_min_p * logits_uncertainty)

		samples = [ _sample_entropix( logits.clone(), temperature, top_k, top_p, min_p, cfg=cfg ) for _ in range(cfg.n_adaptive_samples) ]

		def score_sample(sample):
			one_hot = F.one_hot( sample, logits.shape[-1] )
			log_prob = torch.sum(log_softmax * one_hot)

			confidence_score = (
				(1 - ent) * cfg.ada_score_logits_ent +
				(1 - attn_ent) * cfg.ada_score_attn_ent +
				(1 - vent) * cfg.ada_score_logits_vent +
				(1 - attn_vent) * cfg.ada_score_attn_vent +
				agreement * cfg.ada_score_agree +
				interaction_strength * cfg.ada_score_int
			)

			"""
			if 1024 in sample:
				return 1000
			"""

			return log_prob + confidence_score

		sample_scores = [ score_sample(sample) for sample in samples ]
		best_sample_idx = torch.argmax(torch.asarray(sample_scores))

		res = samples[best_sample_idx]

	"""
	metrics = {
		"attn_entropy": metrics["attn_entropy"],
		"attn_varentropy": metrics["attn_varentropy"],
	}
	"""

	"""
	metrics["temperature"] = temperature
	metrics["top_k"] = top_k
	metrics["top_p"] = top_p
	metrics["min_p"] = min_p
	"""

	return res, metrics