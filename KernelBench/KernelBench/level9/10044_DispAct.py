import torch
import torch.nn as nn
import torch.nn.functional as F


class DispAct(nn.Module):

    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=0.0001, max=10000.0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
