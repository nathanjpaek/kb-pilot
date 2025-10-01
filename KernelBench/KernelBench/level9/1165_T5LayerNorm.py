import torch
from torch import nn


class T5LayerNorm(nn.Module):
    """ Custom LayerNorm for T5 with no mean subtraction and no bias.
    """

    def __init__(self, input_size: 'int', eps: 'float'=1e-05):
        super().__init__()
        self.w = nn.Parameter(torch.ones(input_size))
        self.eps = eps

    def forward(self, x: 'torch.Tensor'):
        x = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.w * x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
