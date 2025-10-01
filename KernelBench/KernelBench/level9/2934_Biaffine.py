import torch
from typing import Callable
from typing import Optional
from torch import nn


class Biaffine(nn.Module):

    def __init__(self, in1_features: 'int', in2_features: 'int',
        out_features: 'int', init_func: 'Optional[Callable]'=None) ->None:
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.linear_in_features = in1_features
        self.linear_out_features = out_features * in2_features
        self._linear = nn.Linear(in_features=self.linear_in_features,
            out_features=self.linear_out_features)
        self.reset_parameters(init_func=init_func)

    def reset_parameters(self, init_func: 'Optional[Callable]'=None) ->None:
        if init_func:
            init_func(self._linear.weight)

    def forward(self, input1: 'torch.Tensor', input2: 'torch.Tensor'):
        batch_size, len1, _dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        affine = self._linear(input1)
        affine = affine.view(batch_size, len1 * self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)
        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2)
        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.
            out_features)
        return biaffine


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in1_features': 4, 'in2_features': 4, 'out_features': 4}]
