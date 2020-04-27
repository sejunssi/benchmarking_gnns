import torch
from torch import nn

def build_vocab(features):
    wordset = set()
    for k, v in features.items():
        for vocab in v:
            wordset.add(int(vocab))
    vocab = {word: i+2 for i, word in enumerate(wordset)}
    vocab['<unk>'] = 0
    vocab['<pad>'] = 1
    return vocab

def feature_embedding(feature, vocab, hidden_dim):
    if len(feature.shape) != 2:
        print(feature)
        return None
    emb = nn.Embedding(len(vocab), hidden_dim)
    return torch.mm(feature.to(torch.float), emb.weight)

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

