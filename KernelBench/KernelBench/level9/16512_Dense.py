import torch
import numpy as np
import torch.nn as nn


def get_einsum_string(ndims, einsum_symbols=None):
    if einsum_symbols is None:
        einsum_symbols = ['u', 'v', 'w', 'x', 'y', 'z']
    assert ndims <= len(einsum_symbols)
    einsum_prefix = ''
    for i in range(ndims):
        einsum_prefix += einsum_symbols[i]
    return einsum_prefix


def maybe_convert_to_list(x):
    if isinstance(x, (int, float)):
        return [x]
    elif isinstance(x, (list, tuple)):
        return list(x)


class Dense(nn.Module):
    """Dense layer."""

    def __init__(self, inp_shape, out_shape, bias=True, reverse_order=False):
        super(Dense, self).__init__()
        self.inp_shape = maybe_convert_to_list(inp_shape)
        self.out_shape = maybe_convert_to_list(out_shape)
        self.reverse_order = reverse_order
        if self.reverse_order:
            self.einsum_str = '...{0},{1}{0}->...{1}'.format(get_einsum_string
                (len(self.inp_shape), ['a', 'b', 'c', 'd']),
                get_einsum_string(len(self.out_shape), ['e', 'f', 'g', 'h']))
            weight_shape = self.out_shape + self.inp_shape
        else:
            self.einsum_str = '...{0},{0}{1}->...{1}'.format(get_einsum_string
                (len(self.inp_shape), ['a', 'b', 'c', 'd']),
                get_einsum_string(len(self.out_shape), ['e', 'f', 'g', 'h']))
            weight_shape = self.inp_shape + self.out_shape
        self.weight = nn.Parameter(torch.zeros(weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_shape))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        fan_in = np.prod(self.inp_shape)
        fan_out = np.prod(self.out_shape)
        std = np.sqrt(1.0 / float(fan_in + fan_out))
        nn.init.normal_(self.weight, std=std)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, inputs):
        output = torch.einsum(self.einsum_str, inputs, self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self):
        return 'inp_shape={}, out_shape={}, bias={}'.format(self.inp_shape,
            self.out_shape, self.bias is not None)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inp_shape': 4, 'out_shape': 4}]
