import torch
import numpy as np
import torch.nn as nn


class DenseParallel(nn.Module):

    def __init__(self, in_features: 'int', out_features: 'int', n_parallel:
        'int', bias: 'bool'=True, device=None, dtype=None) ->None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(DenseParallel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_parallel = n_parallel
        self.weight = nn.Parameter(torch.empty((n_parallel, in_features,
            out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((n_parallel, 1,
                out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return torch.matmul(input, self.weight) + self.bias

    def extra_repr(self) ->str:
        return ('in_features={}, out_features={}, n_parallel={}, bias={}'.
            format(self.in_features, self.out_features, self.n_parallel, 
            self.bias is not None))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4, 'n_parallel': 4}]
