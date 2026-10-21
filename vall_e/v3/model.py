# The v3 prototype: a bidirectional transformer over [text][prompt patches][target patches], where audio enters as
# FSQ values and leaves as one small classifier per scalar (docs/models_v3.md). NAR-demask only for now.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from . import fsq

TEXT, PROMPT, TARGET = 0, 1, 2  # segment ids
NAR_DEMASK, AR, LEN = 0, 1, 2  # mode ids (only NAR_DEMASK is trained in the prototype)


def rotate(x, cos, sin, rot):
	xr, xp = x[..., :rot], x[..., rot:]
	x1, x2 = xr[..., : rot // 2], xr[..., rot // 2 :]
	return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, xp], dim=-1)


class Attention(nn.Module):
	def __init__(self, d, heads, rot_frac, gated, qk_norm):
		super().__init__()
		self.heads, self.head_dim = heads, d // heads
		self.rot = int(self.head_dim * rot_frac) // 2 * 2
		self.qkv = nn.Linear(d, 3 * d, bias=False)
		self.out = nn.Linear(d, d, bias=False)
		# sigmoid output gate per channel (Qwen3.8's gated attention)
		self.gate = nn.Linear(d, d, bias=False) if gated else None
		self.q_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()
		self.k_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()

	def forward(self, x, cos, sin, mask):
		B, L, _ = x.shape
		q, k, v = self.qkv(x).view(B, L, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
		q, k = self.q_norm(q), self.k_norm(k)
		if self.rot:
			q, k = rotate(q, cos, sin, self.rot), rotate(k, cos, sin, self.rot)
		y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, L, -1)
		if self.gate is not None:
			y = y * torch.sigmoid(self.gate(x))
		return self.out(y)


class Block(nn.Module):
	def __init__(self, d, heads, ffn, rot_frac, gated, qk_norm):
		super().__init__()
		self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
		self.attn = Attention(d, heads, rot_frac, gated, qk_norm)
		self.up = nn.Linear(d, 2 * ffn, bias=False)
		self.down = nn.Linear(ffn, d, bias=False)

	def forward(self, x, cos, sin, mask):
		x = x + self.attn(self.n1(x), cos, sin, mask)
		a, b = self.up(self.n2(x)).chunk(2, dim=-1)
		return x + self.down(F.silu(a) * b)


def timestep_embedding(t, dim=256):
	half = dim // 2
	freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
	args = t.float()[:, None] * 1000.0 * freqs[None]
	return torch.cat([args.cos(), args.sin()], dim=-1)


class AdaBlock(nn.Module):
	"""Residual MLP block modulated (shift, scale, gate) by the condition, as in MAR's diffusion head."""
	def __init__(self, width):
		super().__init__()
		self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
		self.mlp = nn.Sequential(nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width))
		self.mod = nn.Linear(width, 3 * width)
		nn.init.zeros_(self.mod.weight); nn.init.zeros_(self.mod.bias)

	def forward(self, x, c):
		shift, scale, gate = self.mod(F.silu(c)).chunk(3, dim=-1)
		return x + gate * self.mlp(self.norm(x) * (1 + scale) + shift)


class FlowHead(nn.Module):
	"""Predicts the rectified-flow velocity of a position's values (all `values` together) at time t, given the
	backbone's hidden state, so a sample is one coherent point rather than independently picked scalars."""
	def __init__(self, d_cond, values, width=512, depth=4, history=False):
		super().__init__()
		self.inp = nn.Linear(values, width)
		if history:
			# the previous patch's values (AR mode), which DiTAR (arXiv 2502.03930) found the head needs to stay
			# coherent and to stop; zero-initialised so a model trained without it is unchanged at first
			self.hist = nn.Linear(values, width)
			self.hist_flag = nn.Embedding(2, width)
			for m in (self.hist, self.hist_flag):
				nn.init.zeros_(m.weight)
			nn.init.zeros_(self.hist.bias)
		self.history = history
		self.cond = nn.Linear(d_cond, width)
		self.time = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
		self.blocks = nn.ModuleList(AdaBlock(width) for _ in range(depth))
		self.final_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
		self.final_mod = nn.Linear(width, 2 * width)
		self.out = nn.Linear(width, values)
		for m in (self.final_mod, self.out):
			nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

	def forward(self, x, t, h, hist=None, has_hist=None):
		"""hist [N, values] previous-patch values and has_hist [N] bool, for a head built with history (None: no history)."""
		c = self.cond(h) + self.time(timestep_embedding(t).to(h.dtype))
		if self.history:
			if hist is None:
				hist = torch.zeros_like(x)
				has_hist = torch.zeros(len(x), dtype=torch.bool, device=x.device)
			c = c + self.hist(hist.to(h.dtype)) * has_hist[:, None].to(h.dtype) + self.hist_flag(has_hist.long()).to(h.dtype)
		x = self.inp(x)
		for block in self.blocks:
			x = block(x, c)
		shift, scale = self.final_mod(F.silu(c)).chunk(2, dim=-1)
		return self.out(self.final_norm(x) * (1 + scale) + shift)


class V3(nn.Module):
	def __init__(self, n_text, d_model=512, layers=12, heads=8, patch=2, rot_frac=1.0, gated=True, qk_norm=True, rope_theta=10000.0,
			head="classifier", flow_width=512, flow_depth=4, ar=False):
		super().__init__()
		self.config = dict(n_text=n_text, d_model=d_model, layers=layers, heads=heads, patch=patch, rot_frac=rot_frac,
			gated=gated, qk_norm=qk_norm, rope_theta=rope_theta, head=head, flow_width=flow_width, flow_depth=flow_depth, ar=ar)
		self.ar = ar
		self.head_type = head
		self.patch = patch
		self.values = patch * fsq.SCALARS
		self.text = nn.Embedding(n_text, d_model)
		self.audio_in = nn.Linear(self.values, d_model)
		self.mask_emb = nn.Parameter(torch.randn(d_model) * 0.02)
		self.segment = nn.Embedding(3, d_model)
		self.mode = nn.Embedding(3, d_model)
		ffn = int(d_model * 8 / 3) // 64 * 64
		self.blocks = nn.ModuleList(Block(d_model, heads, ffn, rot_frac, gated, qk_norm) for _ in range(layers))
		self.norm = nn.RMSNorm(d_model)
		if head == "flow":
			# generates a position's values jointly (see FlowHead)
			self.flow = FlowHead(d_model, self.values, flow_width, flow_depth, history=ar)
		else:
			# one classifier per scalar, padded to the largest level count (invalid levels masked in logits())
			self.head = nn.Linear(d_model, self.values * fsq.MAX_LEVEL)
		if ar:
			# AR mode: a start-of-target embedding, a stop logit per position, and the null condition the flow head
			# sees 10% of the time in training so CFG can run on the head alone (DiTAR)
			self.bos = nn.Parameter(torch.randn(d_model) * 0.02)
			self.stop = nn.Linear(d_model, 1)
			self.null_h = nn.Parameter(torch.zeros(d_model))
		self.register_buffer("level_mask", fsq.level_mask().repeat(patch, 1), persistent=False)  # [values, MAX_LEVEL]
		head_dim = d_model // heads
		rot = int(head_dim * rot_frac) // 2 * 2
		self.register_buffer("inv_freq", 1.0 / rope_theta ** (torch.arange(0, max(rot, 2), 2).float() / max(rot, 2)), persistent=False)

	def forward(self, *args, **kwargs):
		"""Per-scalar logits for every target position, [sum La, values, MAX_LEVEL] (classifier head)."""
		return self.logits(self.encode(*args, **kwargs))

	def encode(self, text, prompt, target, target_masked, mode=NAR_DEMASK):
		"""Lists per sample: text [Lt] ids, prompt [Lp, values] digits, target [La, values] digits, target_masked [La] bool.
		An empty text or prompt drops that condition (CFG). Returns the final hidden state of every target position, [sum La, d_model]."""
		from .data import pack
		return self.encode_packed(pack(text, prompt, target, target_masked), mode)

	def encode_packed(self, b, mode=NAR_DEMASK):
		"""b: the tensors from data.pack(), on this model's device. Returns the target positions' final hidden states."""
		return self.hidden(b, mode)[b["tgt_index"]]

	def hidden(self, b, mode=NAR_DEMASK):
		"""Final hidden states of every position, flattened to [B * L, d_model]. Rows with b["mode"] == AR attend as a
		prefix LM: text and prompt see each other, target positions see the prefix and earlier targets."""
		kind = b["kind"]
		dtype = self.audio_in.weight.dtype
		x = torch.where((kind == 1)[..., None], self.text(b["text_ids"]).to(dtype), self.audio_in(fsq.digits_to_values(b["digits"]).to(dtype)))
		x = torch.where(b["masked"][..., None], self.mask_emb.to(x.dtype), x)
		if self.ar:
			x = torch.where((kind == 4)[..., None], self.bos.to(x.dtype), x)
		modes = b.get("mode")
		mode_emb = self.mode(modes)[:, None, :] if modes is not None else self.mode.weight[mode]
		x = x + self.segment((kind.long() - 1).clamp(0, 2)) + mode_emb
		L = x.shape[1]
		pos = torch.arange(L, device=x.device)
		freqs = pos.float()[:, None] * self.inv_freq[None, :]
		cos, sin = freqs.cos().to(x.dtype), freqs.sin().to(x.dtype)
		keys = (kind > 0)[:, None, :]  # [B, 1, L] key padding
		if modes is not None and bool((modes == AR).any()):
			prefix = b["prefix_len"][:, None, None]
			causal = (pos[None, None, :] < prefix) | (pos[None, None, :] <= pos[None, :, None])  # [B, L, L]
			mask = (keys & (causal | (modes != AR)[:, None, None]))[:, None]
		else:
			mask = keys[:, None]
		for block in self.blocks:
			x = block(x, cos, sin, mask)
		return self.norm(x.flatten(0, 1))

	def logits(self, h):
		logits = self.head(h).view(-1, self.values, fsq.MAX_LEVEL).float()
		return logits.masked_fill(~self.level_mask, float("-inf"))


def ordinal_loss(logits, target, smoothing=0.1):
	"""Cross entropy per scalar against a soft target: 1 - smoothing on the true level, the rest on its valid neighbours,
	so a near miss costs less than a far one. logits [N, values, MAX_LEVEL], target [N, values]."""
	valid = torch.isfinite(logits)
	logp = F.log_softmax(logits, dim=-1).masked_fill(~valid, 0.0)
	soft = F.one_hot(target, fsq.MAX_LEVEL).float() * (1 - smoothing)
	nb = torch.zeros_like(soft)
	for off in (-1, 1):
		idx = target + off
		ok = (idx >= 0) & (idx < fsq.MAX_LEVEL)
		nb.scatter_(-1, idx.clamp(0, fsq.MAX_LEVEL - 1).unsqueeze(-1), ok.float().unsqueeze(-1))
	nb = nb * valid
	soft = soft + smoothing * nb / nb.sum(-1, keepdim=True).clamp(min=1)
	return -(soft * logp).sum(-1).mean()


@torch.no_grad()
def metrics(pred, target):
	"""pred, target [N, values] level indices."""
	hit = pred == target
	group = hit.view(hit.shape[0], -1, len(fsq.LEVELS)).all(-1)  # every scalar of a 1000-way group right
	return dict(scalar_acc=hit.float().mean().item(), group_acc=group.float().mean().item(),
		mean_abs_err=(fsq.digits_to_values(pred) - fsq.digits_to_values(target)).abs().mean().item())


@torch.no_grad()
def demask(model, text, prompt, n_frames, steps=25, cfg=2.5, temperature=1.0):
	"""MaskGIT-style generation of one utterance with the cosine schedule. text [Lt] ids, prompt [Lp, values] digits.
	Returns [n_frames, 32] digits."""
	dev = text.device
	La = math.ceil(n_frames / model.patch)
	digits = torch.zeros(La, model.values, dtype=torch.long, device=dev)
	masked = torch.ones(La, dtype=torch.bool, device=dev)
	for s in range(steps):
		t = (s + 1) / steps
		if cfg > 0:
			empty_t, empty_p = text[:0], prompt[:0]
			out = model([text, empty_t], [prompt, empty_p], [digits, digits], [masked, masked])
			cond, null = out[:La], out[La:]
			logits = null + (cond - null) * cfg
		else:
			logits = model([text], [prompt], [digits], [masked])
		logits = logits.masked_fill(~torch.isfinite(logits), float("-inf"))
		temp = temperature * (1 - t)
		if temp > 0:
			gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-9, 1)))
			sampled = (logits / temp + gumbel).argmax(-1)
		else:
			sampled = logits.argmax(-1)
		logp = F.log_softmax(logits, -1).gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
		conf = logp.mean(-1)  # per patch
		digits = torch.where(masked[:, None], sampled, digits)
		conf = torch.where(masked, conf, torch.full_like(conf, float("inf")))
		n_mask = int(math.floor(math.cos(t * math.pi / 2) * La))
		masked = torch.zeros_like(masked)
		if n_mask > 0:
			masked[conf.topk(n_mask, largest=False).indices] = True
	return digits.view(-1, fsq.SCALARS)[:n_frames]


def snap(values, patch):
	"""[..., patch * 32] values -> nearest level indices on each scalar's grid."""
	levels = fsq._levels.to(values.device).repeat(patch)
	half = (levels // 2).float()
	return torch.minimum((values.float() * half + half).round().clamp(min=0), (levels - 1).float()).long()


def flow_loss(flow, h, target, repeats=4, hist=None, has_hist=None):
	"""Rectified flow: x_t = (1 - t) noise + t data, predict the velocity data - noise. t is logit-normal, and each
	position is drawn `repeats` times per step since the head is cheap next to the backbone. target [N, values] digits;
	hist [N, values] digits and has_hist [N] for a head with history."""
	x1 = fsq.digits_to_values(target).repeat(repeats, 1)
	h = h.repeat(repeats, 1)
	if hist is not None:
		hist, has_hist = fsq.digits_to_values(hist).repeat(repeats, 1), has_hist.repeat(repeats)
	x0 = torch.randn_like(x1)
	t = torch.sigmoid(torch.randn(len(x1), device=x1.device))
	xt = (1 - t[:, None]) * x0 + t[:, None] * x1
	v = flow(xt.to(h.dtype), t, h, hist, has_hist).float()
	return F.mse_loss(v, x1 - x0)


@torch.no_grad()
def flow_sample(flow, h, h_null=None, cfg=0.0, steps=16, temperature=1.0, sway=0.0, hist=None, has_hist=None):
	"""Euler integration from noise (scaled by temperature) to values; CFG mixes the conditional and null velocities.
	sway < 0 spends more of the steps near the noise end, as F5-TTS's sway sampling (s = -1) does.
	h, h_null [N, d]; hist [N, values] values and has_hist [N] for a head with history. Returns [N, values] values."""
	values = flow.out.out_features
	x = torch.randn(len(h), values, device=h.device) * temperature
	grid = torch.linspace(0, 1, steps + 1).tolist()
	if sway:
		grid = [u + sway * (math.cos(math.pi / 2 * u) - 1 + u) for u in grid]
	for k in range(steps):
		t = torch.full((len(h),), grid[k], device=h.device)
		if h_null is not None and cfg > 0:
			hh = torch.cat([hist, hist]) if hist is not None else None
			hf = torch.cat([has_hist, has_hist]) if hist is not None else None
			v = flow(torch.cat([x, x]).to(h.dtype), torch.cat([t, t]), torch.cat([h, h_null]), hh, hf).float()
			v = v[len(h):] + (v[: len(h)] - v[len(h):]) * cfg
		else:
			v = flow(x.to(h.dtype), t, h, hist, has_hist).float()
		x = x + v * (grid[k + 1] - grid[k])
	return x


def reveal_order(n, kind="random", device=None):
	"""Order in which demasking reveals positions: random, or "vdc", a van der Corput (bit-reversed index) order from a
	random offset, so each step's positions are spread evenly over the utterance (the 1-D analogue of Halton ordering,
	arXiv 2503.17076)."""
	if kind == "random":
		return torch.randperm(n, device=device)
	bits = max(1, (n - 1).bit_length())
	rev = sorted(range(n), key=lambda j: int(format(j, f"0{bits}b")[::-1], 2))
	offset = int(torch.randint(n, (1,)))
	return torch.tensor([(j + offset) % n for j in rev], device=device)


@torch.no_grad()
def demask_flow(model, text, prompt, n_frames, steps=25, cfg=2.5, temperature=1.0, flow_steps=8, sway=-1.0,
		cfg_end=None, temperature_end=None, order="random", cfg_mode="backbone"):
	"""MAR-style generation: positions are revealed on the cosine schedule, each sampled from the flow head and snapped
	to the grid. cfg_end / temperature_end anneal linearly over the demask steps (as MAGNeT and MaskGCT do); steps whose
	CFG scale is at most 1 skip the unconditional pass. cfg_mode "head" (models trained with ar=True) guides with the
	flow head's learned null condition, so the backbone runs once per step. Returns [n_frames, 32] digits."""
	dev = text.device
	La = math.ceil(n_frames / model.patch)
	digits = torch.zeros(La, model.values, dtype=torch.long, device=dev)
	masked = torch.ones(La, dtype=torch.bool, device=dev)
	order = reveal_order(La, order, dev)
	done = 0
	for s in range(steps):
		t = (s + 1) / steps
		target_done = La - int(math.floor(math.cos(t * math.pi / 2) * La))
		reveal = order[done:target_done]
		if len(reveal) == 0:
			continue
		frac = s / max(1, steps - 1)
		w = cfg if cfg_end is None else cfg + (cfg_end - cfg) * frac
		temp = temperature if temperature_end is None else temperature + (temperature_end - temperature) * frac
		if w > 1 and cfg_mode == "head":
			hc = model.encode([text], [prompt], [digits], [masked])[reveal]
			hn = model.null_h.to(hc.dtype).expand_as(hc)
		elif w > 1:
			h = model.encode([text, text[:0]], [prompt, prompt[:0]], [digits, digits], [masked, masked])
			hc, hn = h[:La][reveal], h[La:][reveal]
		else:
			hc, hn = model.encode([text], [prompt], [digits], [masked])[reveal], None
		vals = flow_sample(model.flow, hc, hn, cfg=w if w > 1 else 0.0, steps=flow_steps, temperature=temp, sway=sway)
		digits[reveal] = snap(vals, model.patch)
		masked[reveal] = False
		done = target_done
	return digits.view(-1, fsq.SCALARS)[:n_frames]


@torch.no_grad()
def generate_ar(model, text, prompt, max_frames, cfg=1.5, temperature=1.0, flow_steps=8, sway=-1.0, min_frames=0, stop_threshold=0.5):
	"""Streaming-order generation, one patch per step: the backbone gives the new position's hidden state and stop
	probability, and the flow head samples the patch from it and the previous patch, guided on the head alone.
	Recomputes the whole sequence each step (no KV cache yet). Returns [frames, 32] digits."""
	from .data import pack
	dev = text.device
	gen = torch.zeros(0, model.values, dtype=torch.long, device=dev)
	for j in range(math.ceil(max_frames / model.patch)):
		b = pack([text], [prompt], [gen], [torch.zeros(len(gen), dtype=torch.bool, device=dev)], modes=[AR])
		h = model.hidden(b)[b["stop_index"][-1:]]  # the newest position
		if j * model.patch >= min_frames and torch.sigmoid(model.stop(h).float()).item() > stop_threshold:
			break
		hist = fsq.digits_to_values(gen[-1:]) if len(gen) else torch.zeros(1, model.values, device=dev)
		has = torch.tensor([len(gen) > 0], device=dev)
		hn = model.null_h.to(h.dtype).expand_as(h) if cfg > 1 else None
		vals = flow_sample(model.flow, h, hn, cfg=cfg if cfg > 1 else 0.0, steps=flow_steps, temperature=temperature, sway=sway, hist=hist, has_hist=has)
		gen = torch.cat([gen, snap(vals, model.patch)])
	return gen.view(-1, fsq.SCALARS)
