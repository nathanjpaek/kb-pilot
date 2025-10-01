import torch
import torch as th
import torch.nn as nn
from torch.nn.parameter import Parameter


class BilinearMap(nn.Module):

    def __init__(self, nunits):
        super(BilinearMap, self).__init__()
        self.map = Parameter(th.Tensor(nunits, nunits))
        self.nunits = nunits
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.eye_(self.map)

    def forward(self, l, r):
        ncon = l.shape[1]
        r.shape[2]
        nunits = l.shape[3]
        first = th.mm(l.view(-1, nunits), self.map).view(-1, ncon, 1, nunits)
        return th.sum(first * r, dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([16, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nunits': 4}]
