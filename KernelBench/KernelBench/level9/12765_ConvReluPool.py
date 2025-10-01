import torch
from torch.nn import Conv2d
from torch import nn
from torch.nn import functional as F


def Pool(k, stride=1, pad=0):
    return torch.nn.MaxPool2d(k, stride=stride, padding=pad)


class ConvReluPool(nn.Module):

    def __init__(self, fIn, fOut, k, stride=1, pool=2):
        super().__init__()
        self.conv = Conv2d(fIn, fOut, k, stride)
        self.pool = Pool(k)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'fIn': 4, 'fOut': 4, 'k': 4}]
