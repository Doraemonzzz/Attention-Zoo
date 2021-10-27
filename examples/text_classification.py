# change from https://github.com/pbloem/former/blob/master/experiments/classify.py

from torchtext import data, datasets

TEXT = data.Field(lower=True, include_lengths=True)
LABEL = data.Field(sequential=False)
NUM_CLS = 2

train, test = datasets.IMDB.splits(TEXT, LABEL)

print(train)

# def tokenize(label, line):
#     return line.split()

# tokens = []
# for label, line in train_iter:
#     tokens += tokenize(label, line)
