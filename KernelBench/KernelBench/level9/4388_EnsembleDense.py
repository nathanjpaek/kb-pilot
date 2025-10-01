import math
import torch
from torch import nn


class EnsembleDense(nn.Module):
    __constants__ = ['num_ensembles', 'in_features', 'out_features']
    in_features: 'int'
    out_features: 'int'
    weight: 'torch.Tensor'

    def __init__(self, num_ensembles: 'int', in_features: 'int',
        out_features: 'int', bias: 'bool'=True) ->None:
        super(EnsembleDense, self).__init__()
        self.num_ensembles = num_ensembles
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(num_ensembles, in_features,
            out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_ensembles, 1,
                out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        fan = self.in_features
        gain = nn.init.calculate_gain('leaky_relu', param=math.sqrt(5))
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        with torch.no_grad():
            nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return torch.bmm(input, self.weight) + self.bias

    def extra_repr(self) ->str:
        return ('num_ensembles={}, in_features={}, out_features={}, bias={}'
            .format(self.num_ensembles, self.in_features, self.out_features,
            self.bias is not None))


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'num_ensembles': 4, 'in_features': 4, 'out_features': 4}]
