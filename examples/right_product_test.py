import torch
import torch.nn.functional as F

from utils.right_product import causal_product, cross_product

def vanilla_cross_product(q, k, v, eps=1e-9):
	"""[vanilla_cross_product]
	Args:
		q (Tensor): `(N, L, E1)` where L is the target sequence length, N is the batch size,
		E1 is the embedding dimension.
		k (Tensor): `(N, S, E1)` where S is the source sequence length, N is the batch size,
		E1 is the embedding dimension.
		v (Tensor): `(N, S, E2)` where S is the source sequence length, N is the batch size,
		E2 is the embedding dimension.
	"""
	# (N, L, E1) (N, S, E1) -> (N, L, S)
	qk = torch.bmm(q, k.transpose(1, 2))
	# (N, L, S)
	denom = torch.sum(qk, dim=-1, keepdim=True).clamp_min(eps)
	weights = qk / denom
	# (N, L, S) (N, S, E2) -> (N, L, E2)
	output = torch.bmm(weights, v)
	tmp = torch.sum(weights, dim=-1)

	return output

def get_diff(m1, m2):
	diff = torch.norm(m1 - m2)
	print(diff)

def test(batch=1, tgt_len=100, src_len=200, d1=100, d2=200, N=10):
	"""[right product test]
	Args:
		batch (int, optional): [batch size]. Defaults to 1.
		tgt_len (int, optional): [target length]. Defaults to 100.
		src_len (int, optional): [source length]. Defaults to 200.
		d1 (int, optional): [embedding dimension]. Defaults to 100.
		d2 (int, optional): [embedding dimension]. Defaults to 200.
		N (int, optional): [test times]. Defaults to 10.
	"""
	for i in range(N):
		q = torch.exp(torch.randn((batch, tgt_len, d1)))
		k = torch.exp(torch.randn((batch, src_len, d1)))
		v = torch.exp(torch.randn((batch, src_len, d2)))
		# q = torch.randn((batch, tgt_len, d1))
		# k = torch.randn((batch, src_len, d1))
		# v = torch.randn((batch, src_len, d2))
		o1 = vanilla_cross_product(q, k, v)
		o2 = cross_product(q, k, v)
		get_diff(o1, o2)
		
def main():
	batch = 1
	tgt_len = 100
	src_len = 200
	d1 = 100
	d2 = 200
	N = 10

	# test(1, 5, 4, 2, 3)
	test(batch, tgt_len, src_len, d1, d2, N)

if __name__ == "__main__":
	main()