import torch
import torch.nn as nn


class ScalingBlock(nn.Module):

    def __init__(self, temp=5.0, **kwargs):
        super(ScalingBlock, self).__init__()
        self.temp = temp

    def forward(self, x):
        x = x / self.temp
        return x

    def extra_repr(self):
        return 'temp=%.3e' % self.temp


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
