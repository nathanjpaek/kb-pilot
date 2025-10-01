import torch
import torch.nn as nn
import torch.fx


class ConvGelu(torch.nn.Module):

    def __init__(self):
        super(ConvGelu, self).__init__()
        self.conv = nn.Conv2d(3, 32, 3, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.gelu(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
