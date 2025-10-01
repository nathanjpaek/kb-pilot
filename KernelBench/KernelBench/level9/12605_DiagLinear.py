import math
import torch
from torch import Tensor
from torch import nn


class DiagLinear(nn.Module):
    """Applies a diagonal linear transformation to the incoming data: :math:`y = xD^T + b`"""
    __constants__ = ['features']

    def __init__(self, features, bias=True):
        super(DiagLinear, self).__init__()
        self.features = features
        self.weight = nn.Parameter(Tensor(features))
        if bias:
            self.bias = nn.Parameter(Tensor(features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            bound = 1 / math.sqrt(self.features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        output = input.mul(self.weight)
        if self.bias is not None:
            output += self.bias
        return output

    def extra_repr(self):
        return 'features={}, bias={}'.format(self.features, self.bias is not
            None)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'features': 4}]
