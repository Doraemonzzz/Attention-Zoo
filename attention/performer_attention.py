import torch
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from typing import Optional
from torch import nn
from utils.right_product import causal_product, cross_product

class PerformerAttention(nn.Module):
	"""[summary]
	performer attention in "Rethinking Attention with Performers
	https://arxiv.org/abs/2009.14794
	"""
	def __init__(
		self,
		embed_dim,
		num_heads,
		kdim=None,
		vdim=None,
		proj_dim=None,
		sigma=1.0,
		dropout_rate=0.0,
		causal=False,
	):
		super().__init__()
		self.embed_dim = embed_dim
		self.kdim = kdim if kdim is not None else embed_dim
		self.vdim = vdim if kdim is not None else embed_dim
		self.num_heads = num_heads
		# q, k, v projection
		self.k_proj = nn.Linear(self.kdim, embed_dim)
		self.v_proj = nn.Linear(self.vdim, embed_dim)
		self.q_proj = nn.Linear(embed_dim, embed_dim)
		# outprojection
		self.out_proj = nn.Linear(embed_dim, embed_dim)
		# dropout rate
		self.dropout_rate = dropout_rate
		# causal
		self.causal = causal
		# rfa
		self.sigma = sigma
		self.proj_dim = proj_dim if proj_dim is not None else embed_dim
		# (H, D, E)
		head_dim = embed_dim // num_heads
		self.random_matrices = self.sigma * torch.randn(self.num_heads, self.proj_dim, head_dim)

		assert (self.embed_dim % self.num_heads == 0), "embed_dim must be divisible by num_heads"

	def forward(
		self,
		query: Tensor,
		key: Optional[Tensor],
		value: Optional[Tensor],
		attn_mask: Optional[Tensor] = None,
	):
		"""Input shape: Sequence x Batch x Embedding
		Args:
			query (Tensor): `(L, N, E1)` where L is the target sequence length, N is the batch size,
			E1 is the embedding dimension.
			key (Tensor): `(S, N, E1)` where S is the source sequence length, N is the batch size,
			E1 is the embedding dimension.
			value (Tensor): `(S, N, E2)` where S is the source sequence length, N is the batch size,
			E2 is the embedding dimension.
			attn_mask (Optional[Tensor], optional): typically used to implement causal attention, 
			where the mask prevents the attention from looking forward in time (default: None).
		"""
		num_heads = self.num_heads
		tgt_len, bsz, embed_dim = query.size()
		src_len = key.size(0)
		head_dim = embed_dim // num_heads

		# get q, k, v
		# (L, N, E)
		q = self.q_proj(query)
		# (S, N, E)
		k = self.k_proj(key)
		# (S, N, E)
		v = self.v_proj(value)

		# multihead
		# (L, N, h, d)
		q = q.view(-1, bsz, num_heads, head_dim)
		# (S, N, h, d)
		k = k.view(-1, bsz, num_heads, head_dim)
		# (N * h, S, d)
		v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

		# feature transform
		phi_q = self.random_project(q, self.random_matrices)
		phi_k = self.random_project(k, self.random_matrices)

		# (N * h, L, d)
		if self.causal:
			attn_output = causal_product(phi_q, phi_k, v)
		else:
			attn_output = cross_product(phi_q, phi_k, v)
		# output
		# L, N, E
		attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
		# L, N, E
		attn_output = self.out_proj(attn_output)

		return attn_output

	def random_project(
		self,
		x: Tensor,
		random_matrices: Tensor,
	):
		"""[random projection]
		Args:
			x (Tensor): `(L, N, H, E)` where L is the target sequence length, N is the batch size,
			H is the number of heads, E is the embedding dimension.
			random_matrices (Tensor): `(H, D, E)` where H is the number of heads, D is the projection dimension,
			E is the embedding dimension.
		"""
		L, N, H, E = x.size()
		H, D, E = random_matrices.size()
		scale = 1 / np.sqrt(2 * D)
		# l2 normalize under feature dimension
		# (L, N, H, E)
		x_normalize = F.normalize(x, p=2, dim=-1)
		# feature transform
		# (L, N, H, E) (H, D, E) -> (L, N, H, D)
		x_transform = torch.einsum("lnhe,hde->lnhd", x_normalize, random_matrices)
		# get cos, sin
		# (L, N, H, D)
		x_exp_pos = torch.exp(x_transform)
		x_exp_neg = torch.exp(-x_transform)
		# (L, N, H, D) (L, N, H, D) -> (L, N, H, 2 * D) -> (N * H, L, 2 * D)
		phi_x = scale * torch.cat([x_exp_pos, x_exp_neg], dim=-1).contiguous().view(-1, L, 2 * D)

		return phi_x