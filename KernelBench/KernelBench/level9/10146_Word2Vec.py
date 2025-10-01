import torch
from torch import nn
import torch.functional as F
import torch.nn.functional as F


class Word2Vec(torch.nn.Module):

    def __init__(self, vocab_size, embedding_size=300):
        super(Word2Vec, self).__init__()
        self.E = nn.Linear(vocab_size, embedding_size, bias=False)
        self.W = nn.Linear(embedding_size, vocab_size)

    def forward(self, one_hot):
        z_e = self.E(one_hot)
        z_w = self.W(z_e)
        return F.log_softmax(z_w, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'vocab_size': 4}]
