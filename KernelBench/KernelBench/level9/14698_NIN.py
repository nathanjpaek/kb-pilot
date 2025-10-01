import string
import torch
import numpy as np
import torch.utils.data
import torch
import torch.nn as nn


def _einsum(a, b, c, x, y):
    einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
    return torch.einsum(einsum_str, x, y)


def contract_inner(x, y):
    """tensordot(x, y, 1)."""
    x_chars = list(string.ascii_lowercase[:len(x.shape)])
    y_chars = list(string.ascii_lowercase[len(x.shape):len(y.shape) + len(x
        .shape)])
    y_chars[0] = x_chars[-1]
    out_chars = x_chars[:-1] + y_chars[1:]
    return _einsum(x_chars, y_chars, out_chars, x, y)


def variance_scaling(scale, mode, distribution, in_axis=1, out_axis=0,
    dtype=torch.float32, device='cpu'):

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis
            ]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == 'fan_in':
            denominator = fan_in
        elif mode == 'fan_out':
            denominator = fan_out
        elif mode == 'fan_avg':
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
                'invalid mode for variance scaling initializer: {}'.format(
                mode))
        variance = scale / denominator
        if distribution == 'normal':
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(
                variance)
        elif distribution == 'uniform':
            return (torch.rand(*shape, dtype=dtype, device=device) * 2.0 - 1.0
                ) * np.sqrt(3 * variance)
        else:
            raise ValueError(
                'invalid distribution for variance scaling initializer')
    return init


def default_init(scale=1.0):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, 'fan_avg', 'uniform')


class NIN(nn.Module):

    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(default_init(scale=init_scale)((in_dim,
            num_units)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        y = contract_inner(x, self.W) + self.b
        return y.permute(0, 3, 1, 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'num_units': 4}]
