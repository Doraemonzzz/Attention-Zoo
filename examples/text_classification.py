# change from https://github.com/pbloem/former/blob/master/experiments/classify.py

import torch
import torch.nn.functional as F

from torchtext import data, datasets
from attention.vanilla_attention import VanillaAttention
from attention.linear_attention import LinearAttention
from transformer.transformer import TransformerClassifier
from tqdm import tqdm
from argparse import ArgumentParser

def get_attention_model(attention_type):
	if attention_type == "vanilla":
		return VanillaAttention
	elif attention_type == "linear":
		return LinearAttention

def get_acc(model, data):
	total_cnt = 0
	correct_cnt = 0
	for batch in tqdm(data):
		input = batch.text[0]
		output = model(input)
		label = batch.label - 1
		pred_label = output.argmax(-1)
		# get res
		total_cnt += input.size(1)
		correct_cnt += (pred_label == label).sum().item()

		break
	
	return total_cnt / correct_cnt

def main():
	# device = torch.device("cuda:0" if torch.cuda.is_avaliable() else "cpu")
	device = "cpu"
	TEXT = data.Field(lower=True, include_lengths=True)
	LABEL = data.Field(sequential=False)

	# data parameter
	vocab_size = 20000
	num_classes = 2

	# parameter
	parser = ArgumentParser()
	parser.add_argument("--attention-type", 
						dest="attention_type",
						default="vanilla")
	parser.add_argument("--embed-dim", 
						dest="embed_dim",
						default=768,
						type=int)
	parser.add_argument("--ffn-embed-dim", 
						dest="ffn_embed_dim",
						default=768,
						type=int)
	parser.add_argument("--num-heads", 
						dest="num_heads",
						default=8,
						type=int)
	parser.add_argument("--num-layers", 
						dest="num_layers",
						default="1",
						type=int)

	parser.add_argument("--num-epochs", 
						dest="num_epochs",
						default="10",
						type=int)
	parser.add_argument("--lr", 
						dest="lr",
						default=0.001,
						type=float)
	parser.add_argument("--max-seqlen", 
						dest="max_seqlen",
						default=500,
						type=int)
	parser.add_argument("--batch-size", 
						dest="batch_size",
						default=16,
						type=int)
	parser.add_argument("--test-freq", 
						dest="test_freq",
						default=1,
						type=int)

	args = parser.parse_args()
	# # model parameter
	# attention_type = "vanilla"
	# embed_dim = 768
	# ffn_embed_dim = 768
	# num_heads = 8
	# num_layers = 1

	# # training parameter
	# num_epochs = 10
	# lr = 0.001
	# max_seqlen = 500
	# batch_size = 16
	# test_freq = 1

	# model parameter
	attention_type = args.attention_type
	embed_dim = args.embed_dim
	ffn_embed_dim = args.ffn_embed_dim
	num_heads = args.num_heads
	num_layers = args.num_layers

	# training parameter
	num_epochs = args.num_epochs
	lr = args.lr
	max_seqlen = args.max_seqlen
	batch_size = args.batch_size
	test_freq = args.test_freq

	# data
	train, test = datasets.IMDB.splits(TEXT, LABEL)
	TEXT.build_vocab(train, max_size=vocab_size)
	LABEL.build_vocab(train)
	train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=batch_size, device=device)
	num_tokens = len(TEXT.vocab)

	# model
	attention = get_attention_model(attention_type)
	model = TransformerClassifier(
				embed_dim,
				ffn_embed_dim,
				num_heads,
				attention,
				num_layers,
				num_tokens,
				num_classes,
				max_seqlen
			)

	# opt
	optimier = torch.optim.Adam(lr=lr, params=model.parameters())
	criterion = F.cross_entropy

	for i in range(num_epochs):
		print(f"epoch {i + 1}")
		model.train()
		
		total_cnt = 0
		correct_cnt = 0
		for batch in tqdm(train_iter):
			optimier.zero_grad()
			# batch.text: (L, N)
			# label: (1, 2)
			input = batch.text[0]
			# 防止超过最大长度
			if input.size(0) > max_seqlen:
				input = input[:max_seqlen]
			label = batch.label - 1
			output = model(input)

			loss = criterion(output, label)
			loss.backward()
			optimier.step()

			break

		if i % test_freq == 0:
			acc = get_acc(model, test_iter)
			print(f"In epoch {i + 1}, test accuracy is {correct_cnt}")

if __name__ == "__main__":
	main()
