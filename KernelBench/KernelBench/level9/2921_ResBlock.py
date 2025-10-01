import torch
import torch.utils.data
import torch
import torch.nn as nn


class ResBlock(nn.Module):

    def __init__(self, inFe):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inFe, inFe, 3, 1, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(inFe, inFe, 3, 1, 1)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        x = x + res
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inFe': 4}]
