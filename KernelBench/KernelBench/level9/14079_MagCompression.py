import torch
from torch import Tensor
from torch import nn
from torch.nn.parameter import Parameter


class MagCompression(nn.Module):

    def __init__(self, n_freqs: 'int', init_value: 'float'=0.3):
        super().__init__()
        self.c: 'Tensor'
        self.register_parameter('c', Parameter(torch.full((n_freqs,),
            init_value), requires_grad=True))
        self.mn: 'Tensor'
        self.register_parameter('mn', Parameter(torch.full((n_freqs,), -0.2
            ), requires_grad=True))

    def forward(self, x: 'Tensor'):
        x = x.pow(self.c) + self.mn
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_freqs': 4}]
