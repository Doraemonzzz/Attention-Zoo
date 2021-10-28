import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Optional
from torch import nn

def causal_product(q, k, v, eps=1e-6):
	"""[right product of causal attention]
	Args:
		q (Tensor): `(N, L, E1)` where L is the target sequence length, N is the batch size,
		E1 is the embedding dimension.
		k (Tensor): `(N, L, E1)` where L is the source sequence length, N is the batch size,
		E1 is the embedding dimension.
		v (Tensor): `(N, L, E2)` where L is the source sequence length, N is the batch size,
		E2 is the embedding dimension.
	"""
	# (N, L, E1) (N, L, E2) -> (N, L, E1, E2)
	kv = torch.einsum("nle,nld->nled", k, v)
	# (N, L, E1, E2) -> (N, L, E1, E2)
	kv_cum = torch.cumsum(kv, dim=1)
	# (N, L, E1) (N, L, E1, E2) -> (N, L, E2)
	qkv = torch.einsum("nle,nled->nld", q, kv_cum)
	# (N, L, E1) -> (N, L, E1)
	k_cum = torch.cumsum(k, dim=1)
	# (N, L, E1) (N, L, E1) -> (N, L)
	denom = torch.clamp_min(torch.einsum("nle,nle->nl", q, k_cum), eps)
	# (N, L, E2) (N, L) -> (N, L, E2)
	attn_output_weights = qkv / denom

	return attn_output_weights

def cross_product(q, k, v, eps=1e-6):
	"""[right product of cross attention]
	Args:
		q (Tensor): `(N, L, E1)` where L is the target sequence length, N is the batch size,
		E1 is the embedding dimension.
		k (Tensor): `(N, S, E1)` where S is the source sequence length, N is the batch size,
		E1 is the embedding dimension.
		v (Tensor): `(N, S, E2)` where S is the source sequence length, N is the batch size,
		E2 is the embedding dimension.
	"""
	# (N, S, E1) (N, S, E2) -> (N, E1, E2)
	kv = torch.einsum("nle,nld->ned", k, v)
	# (N, L, E1) (N, E1, E2) -> (N, L, E2)
	qkv = torch.einsum("nle,ned->nld", q, kv)
	# (N, L, E1) (N, E1) -> (N, L)
	denom = torch.clamp_min(torch.einsum("nle,nd->nl", q, torch.sum(k, axis=1)), eps)
	# (N, L, E2) (N, L) -> (N, L, E2)
	attn_output_weights = qkv / denom

	return attn_output_weights