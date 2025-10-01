import torch
import torch.nn as nn


class Simple224Upsample(nn.Module):

    def __init__(self, arch=''):
        super(Simple224Upsample, self).__init__()
        self.upsample = nn.Upsample(mode='nearest', scale_factor=7)
        self.arch = arch

    def forward(self, x):
        return self.upsample(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
