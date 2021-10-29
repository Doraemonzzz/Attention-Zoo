import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Optional
from torch import nn
from utils.right_product import causal_product, cross_product

class LinearAttention(nn.Module):
	"""[summary]
	linear attention in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
	https://arxiv.org/abs/2006.16236
	"""
	def __init__(
		self,
		embed_dim,
		num_heads,
		kdim=None,
		vdim=None,
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

		assert (self.embed_dim % self.num_heads == 0), "embed_dim must be divisible by num_heads"

	def forward(
		self,
		query: Optional[Tensor],
		key: Optional[Tensor],
		value: [Tensor],
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
		# (N * h, L, d)
		q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
		# (N * h, S, d)
		k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
		v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

		# (N * h, L, d)
		if self.causal:
			attn_output = causal_product(q, k, v)
		else:
			attn_output = cross_product(q, k, v)
		# output
		# L, N, E
		attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
		# L, N, E
		attn_output = self.out_proj(attn_output)

		return attn_output
