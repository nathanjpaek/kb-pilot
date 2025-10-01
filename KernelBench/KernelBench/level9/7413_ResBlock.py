import torch
from torch import nn
import torch.nn.functional as F


class LinearAndMultiply(nn.Module):

    def __init__(self, input_size, output_size, use_multiply=True,
        linear_block=nn.Linear):
        super().__init__()
        self._activation = nn.CELU()
        self._linear = linear_block(input_size, output_size)
        self._use_multiply = use_multiply
        if self._use_multiply:
            self._to_multiplier = linear_block(output_size, output_size)

    def forward(self, x, *extra):
        x = self._activation(self._linear(x, *extra))
        if not self._use_multiply:
            return x
        return x * torch.tanh(self._to_multiplier(x, *extra))


class ResBlock(nn.Module):

    def __init__(self, input_size, output_size, use_multiply=True,
        linear_block=nn.Linear, use_norm=True):
        super().__init__()
        self._linear_block = LinearAndMultiply(input_size, output_size,
            use_multiply=False, linear_block=linear_block)
        self._mul_block = LinearAndMultiply(output_size, output_size,
            use_multiply=use_multiply, linear_block=linear_block)
        self._use_norm = use_norm
        if self._use_norm:
            self._norm = nn.LayerNorm(output_size)
        self._pad_size = output_size - input_size
        assert self._pad_size >= 0

    def forward(self, x, *extra):
        padded_input = F.pad(x, (0, self._pad_size))
        x = self._mul_block(self._linear_block(x, *extra), *extra)
        x = padded_input + x
        if self._use_norm:
            x = self._norm(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]
