import torch
import torch as th
import torch.nn as nn


class EuclideanDistance(nn.Module):

    def __init__(self):
        super(EuclideanDistance, self).__init__()
        self.m = nn.Sigmoid()

    def forward(self, i, j):
        i_norm = self.m(i)
        j_norm = self.m(j)
        return th.sqrt(th.sum((i_norm - j_norm) ** 2, dim=-1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
