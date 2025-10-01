import torch
from torch import nn


class IdfCombination(nn.Module):

    def forward(self, scores, idf):
        idf = idf.softmax(dim=1)
        return (scores * idf).sum(dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
