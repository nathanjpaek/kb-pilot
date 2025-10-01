import torch
import torch.optim
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, in_f, out_f):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(in_f, out_f, 1, 1, padding=0, bias=False)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_f': 4, 'out_f': 4}]
