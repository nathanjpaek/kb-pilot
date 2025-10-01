import torch
import torch.nn as nn
import torch.optim


class LayerNorm(nn.Module):
    """A Layer Normalization layer.
       Lei Ba, Jimmy, Jamie Ryan Kiros, and Geoffrey E. Hinton.
       arXiv preprint arXiv:1607.06450 (2016).
    """

    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, dim))
        self.gain = nn.Parameter(torch.ones(1, dim))

    def forward(self, x):
        out = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-06
            )
        return self.gain * out + self.bias


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
