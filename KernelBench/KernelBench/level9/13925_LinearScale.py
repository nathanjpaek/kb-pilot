import torch
from torch import nn


class LinearScale(nn.Module):

    def __init__(self, scale, bias):
        super(LinearScale, self).__init__()
        self.scale_v = scale
        self.bias_v = bias
        pass

    def forward(self, x):
        out = x * self.scale_v + self.bias_v
        return out

    def __repr__(self):
        repr = (
            f'{self.__class__.__name__}(scale_v={self.scale_v},bias_v={self.bias_v})'
            )
        return repr


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'scale': 1.0, 'bias': 4}]
