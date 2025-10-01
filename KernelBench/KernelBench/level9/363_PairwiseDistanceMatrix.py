import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


class PairwiseDistanceMatrix(nn.Module):

    def __init__(self):
        super(PairwiseDistanceMatrix, self).__init__()

    def forward(self, X, Y):
        X2 = (X ** 2).sum(1).view(-1, 1)
        Y2 = (Y ** 2).sum(1).view(1, -1)
        D = X2 + Y2 - 2.0 * torch.mm(X, Y.t())
        return D


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
