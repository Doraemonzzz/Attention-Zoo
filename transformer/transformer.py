import torch
import torch.nn.functional as F

from torch import nn
from .transformer_layer import TransformerEncoderLayer

class TransformerClassifier(nn.Module):
	"""
	Transformer for classifying sequences
	"""
	def __init__(
		self,
		embed_dim,
		ffn_embed_dim,
		num_heads,
		attention,
		num_layers,
		num_tokens,
		num_classes,
		max_seqlen,
	):
		"""
		Args:
			embed_dim (int): embedding dim
			ffn_embed_dim (int): ffn embedding dim
			num_heads (int): number of heads
			attention (class): attention module
			num_layers (int): number of transformer layer
			num_tokens (int): number of tokens
			num_classes (int): number of classes
			max_seqlen (int): max length of sequence length
		"""
		super().__init__()

		self.token_embedding = nn.Embedding(num_embeddings=num_tokens, embedding_dim=embed_dim)
		self.pos_embedding = nn.Embedding(num_embeddings=max_seqlen, embedding_dim=embed_dim)
		transformer_blocks = []
		for i in range(num_layers):
			transformer_blocks.append(TransformerEncoderLayer(embed_dim, ffn_embed_dim, num_heads, attention))
		self.transformer_blocks = nn.Sequential(transformer_blocks)
		self.fc = nn.Linear(embed_dim, num_classes)

	def forward(self, x):
		"""
		Args:
			x (Tensor): token index, `(L, N)`
		"""
		# (L, N, E)
		token_embedding = self.token_embedding(x)
		# (L)
		pos_embedding = self.pos_embedding(torch.range(x.shape[0], device=x.device))
		# add
		# (L, N, E)
		x = token_embedding + pos_embedding
		# (L, N, E)
		x = self.transformer_blocks(x)
		# (L, N, num_classes)
		x = self.fc(x)

		return x
