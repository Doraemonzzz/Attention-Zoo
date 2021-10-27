# change from https://github.com/pbloem/former/blob/master/experiments/classify.py

import torch
import torch.nn.functional as F

from torchtext import data, datasets
from vanilla_attention.vanilla_attention import VanillaAttention
from transformer.transformer import TransformerClassifier
from tqdm import tqdm

TEXT = data.Field(lower=True, include_lengths=True)
LABEL = data.Field(sequential=False)
vocab_size = 20000
batch_size = 16
# device = torch.device("cuda:0" if torch.cuda.is_avaliable() else "cpu")
device = "cpu"
attention_type = "vanilla"
NUM_CLS = 2
embed_dim = 768
ffn_embed_dim = 768
num_heads = 8
num_layers = 1
num_epochs = 1
lr = 0.001


train, test = datasets.IMDB.splits(TEXT, LABEL)

TEXT.build_vocab(train, max_size=vocab_size)
LABEL.build_vocab(train)

train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=batch_size, device=device)

if attention_type == "vanilla":
	attention = VanillaAttention

num_tokens = len(TEXT.vocab)
num_classes = 2
# print(LABEL.vocab)
max_seqlen = 500

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

		# print(output.shape)
		# print(label.shape)
		loss = criterion(output, label)
		loss.backward()
		optimier.step()
		# print(loss)
		pred_label = output.argmax(-1)

		total_cnt += input.size(1)
		correct_cnt += (pred_label == label).sum().item()
		# print(batch.text[0].shape)
		# print(batch.label.shape)
		# print(batch.label)
		break
		# out = model()
	print(correct_cnt / total_cnt)



print(len(TEXT.vocab))
print(len(LABEL.vocab))

# print(vars(train.examples[0]))

# def tokenize(label, line):
#     return line.split()

# tokens = []
# for label, line in train_iter:
#     tokens += tokenize(label, line)
