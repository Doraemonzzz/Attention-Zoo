# change from https://github.com/pytorch/fairseq/blob/main/fairseq/modules/multihead_attention.py

import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Optional
from torch import nn

class VanillaAttention(nn.Module):
	"""[summary]
	vanilla attention in "Attention Is All You Need"
	https://arxiv.org/abs/1706.03762
	"""
	def __init__(
		self,
		embed_dim,
		num_heads,
		kdim=None,
		vdim=None,
		dropout_rate=0.0,
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
		# (N * h, L, d)
		q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
		# (N * h, S, d)
		k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
		v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

		# scaling
		scaling = float(head_dim) ** -0.5
		q *= scaling

		# (N * h, L, S)
		attn_output_weights = torch.bmm(q, k.transpose(1, 2))

		# mask
		# change mask to 3d
		if attn_mask is not None:
			if attn_mask.dim() == 3:
				attn_mask = attn_mask.unsqueeze(0)
				assert (list(attn_mask.size()) == [1, tgt_len, src_len]), \
						"The size of the 2D attn_mask is not correct."
			elif attn_mask.dim() == 3:
				assert (list(attn_mask.size()) == [bsz * num_heads, tgt_len, src_len]), \
						"The size of the 3D attn_mask is not correct."
			else:
				assert False, f"Tattn_mask's dimension {attn_mask.size()} is not supported."

		if attn_mask is not None:
			attn_output_weights += attn_mask

		# softmax
		# (N * h, L, S)
		attn_output_weights = F.softmax(attn_output_weights, dim=-1)
		# dropout
		attn_output_weights = F.dropout(attn_output_weights, p=self.dropout_rate, training=self.training)
		# output
		# (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
		attn_output = torch.bmm(attn_output_weights, v)
		# L, N, E
		attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
		# L, N, E
		attn_output = self.out_proj(attn_output)

		return attn_output