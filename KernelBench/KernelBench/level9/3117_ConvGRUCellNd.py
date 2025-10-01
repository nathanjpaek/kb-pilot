import torch
import torch.nn as nn
import torch.jit
import torch.nn


class ConvGRUCellNd(nn.Module):

    def __init__(self, in_size, out_size, kernel_size, N=1, **kwargs):
        super(ConvGRUCellNd, self).__init__()
        conv = eval(f'nn.Conv{N}d')
        self.conv_ir = conv(in_size, out_size, kernel_size, **kwargs)
        self.conv_hr = conv(in_size, out_size, kernel_size, **kwargs)
        self.conv_iz = conv(in_size, out_size, kernel_size, **kwargs)
        self.conv_hz = conv(in_size, out_size, kernel_size, **kwargs)
        self.conv_in = conv(in_size, out_size, kernel_size, **kwargs)
        self.conv_hn = conv(in_size, out_size, kernel_size, **kwargs)

    def forward(self, inputs, state):
        r = torch.sigmoid(self.conv_ir(inputs) + self.conv_hr(state))
        z = torch.sigmoid(self.conv_iz(inputs) + self.conv_hz(state))
        n = torch.tanh(self.conv_in(inputs) + self.conv_hn(state * r))
        return z * state + (1 - z) * n


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4, 'out_size': 4, 'kernel_size': 4}]
