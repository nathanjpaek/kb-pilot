import torch
from torch import nn
import torch.utils.data
import torch.nn.parallel


class disparityentropy(nn.Module):

    def __init__(self, maxdisp):
        super(disparityentropy, self).__init__()

    def forward(self, x):
        out = torch.sum(-x * torch.log(x), 1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'maxdisp': 4}]
