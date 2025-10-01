import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):

    def __init__(self, input_size, hidden_size, kernel_size, stride=1,
        dilation=1, groups=1, bias=True, sigmoid=None, tanh=None):
        self.left_padding = (kernel_size - 1) * dilation
        super(CausalConv1d, self).__init__(input_size, hidden_size,
            kernel_size, stride=stride, padding=0, dilation=dilation,
            groups=groups, bias=bias)

    def forward(self, input):
        x = F.pad(input.permute(1, 2, 0), (self.left_padding, 0))
        conv_out = super(CausalConv1d, self).forward(x)
        return conv_out.permute(2, 0, 1)


class Wave(nn.Module):

    def __init__(self, input_size, hidden_size, layers=3, activation='tanh'):
        super(Wave, self).__init__()
        self.layers = []
        prev_size = input_size
        for layer in range(layers):
            conv = CausalConv1d(prev_size, hidden_size, kernel_size=2,
                dilation=2 ** layer)
            self.layers.append(conv)
            self.add_module('layer' + str(layer), conv)
            prev_size = hidden_size

    def forward(self, data):
        for layer in self.layers:
            data = layer(data)
        return data


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
