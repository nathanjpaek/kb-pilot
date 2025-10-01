import torch
import torch.nn as nn
import torch.nn.functional as F


class Word2Vec(nn.Module):

    def __init__(self, vocabulary_size, embedding_size):
        super(Word2Vec, self).__init__()
        self.w1 = nn.Parameter(torch.randn(vocabulary_size, embedding_size,
            requires_grad=True))
        self.w2 = nn.Parameter(torch.randn(embedding_size, vocabulary_size,
            requires_grad=True))

    def forward(self, one_hot):
        z1 = torch.matmul(one_hot, self.w1)
        z2 = torch.matmul(z1, self.w2)
        log_softmax = F.log_softmax(z2, dim=1)
        return log_softmax


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'vocabulary_size': 4, 'embedding_size': 4}]
