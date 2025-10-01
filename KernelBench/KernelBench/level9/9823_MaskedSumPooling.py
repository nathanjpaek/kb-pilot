import torch
from torch import nn


class MaskedSumPooling(nn.Module):

    def __init__(self):
        super(MaskedSumPooling, self).__init__()

    def forward(self, embedding_matrix):
        return torch.sum(embedding_matrix, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
