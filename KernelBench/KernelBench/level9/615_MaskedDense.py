from torch.nn import Module
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules import Module


class MaskedDense(Module):

    def __init__(self, in_dim, out_dim, bias=False):
        super(MaskedDense, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = bias
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available(
            ) else torch.FloatTensor
        self.weight = nn.Parameter(self.floatTensor(out_dim, in_dim),
            requires_grad=True)
        if bias:
            self.bias = nn.Parameter(self.floatTensor(out_dim),
                requires_grad=True)
        else:
            self.bias = None
        self.mask = nn.Parameter(self.floatTensor(out_dim, in_dim),
            requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        init.ones_(self.mask)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, bias=self.bias)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
