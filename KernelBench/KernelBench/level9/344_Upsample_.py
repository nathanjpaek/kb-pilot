import torch
from torch import nn


class Upsample_(nn.Module):

    def __init__(self, scale=2):
        super(Upsample_, self).__init__()
        self.upsample = nn.Upsample(mode='bilinear', scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
