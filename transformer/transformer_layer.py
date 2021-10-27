import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from typing import Optional

class TransformerEncoderLayer(nn.Module):
	def __init__(
		self, 
		encoder_embed_dim,
		encoder_ffn_embed_dim,
		num_heads,
		attention,
		dropout_rate=0.0,
	):
		super().__init__()
		self.embed_dim = encoder_embed_dim
		self.num_heads = num_heads
		self.encoder_ffn_embed_dim = encoder_ffn_embed_dim
		self.dropout_rate = dropout_rate

		self.attention = attention(self.embed_dim, self.num_heads)
		self.fc1 = nn.Linear(self.embed_dim, self.encoder_ffn_embed_dim)
		self.fc2 = nn.Linear(self.encoder_ffn_embed_dim, self.embed_dim)
		self.layernorm = nn.LayerNorm(self.embed_dim)

	def forward(
		self, 
		x: Tensor, 
		attn_mask: Optional[Tensor] = None
	):
		"""Compute encoder forward:
		Args:
			x (Tensor): `(L, N, E)` where L is the target sequence length, N is the batch size,
			E is the embedding dimension.
			attn_mask (ByteTensor, optional): `(L, S)` where L is the target sequence length,
			S is the source sequence length.
			attn_mask[i, j] = 1 means this position should be used for computing.
		"""
		# attention
		if attn_mask is not None:
			attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

		residual = x

		x = self.layernorm(x)
		x = self.attention(
			query=x,
			key=x,
			value=x,
			attn_mask=attn_mask,
		)
		x = F.dropout(x, p=self.dropout_rate, training=self.training)
		x += residual

		# MLP
		residual = x
		x = self.layernorm(x)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, p=self.dropout_rate, training=self.training)
		x = self.fc2(x)
		x = F.dropout(x, p=self.dropout_rate, training=self.training)
		x += residual

		return x