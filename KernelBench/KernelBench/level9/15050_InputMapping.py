import math
import torch
import torch.fft
import torch.nn


class InputMapping(torch.nn.Conv1d):

    def __init__(self, in_channels: 'int', out_channels: 'int', omega_0:
        'float', stride: 'int'=1, bias: 'bool'=True):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
            kernel_size=1, stride=stride, padding=0, bias=bias)
        self.omega_0 = omega_0
        self.weight.data.normal_(0.0, 2 * math.pi * self.omega_0)

    def forward(self, x):
        out = super().forward(x)
        out = torch.cat([torch.cos(out), torch.sin(out)], dim=1)
        return out


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'omega_0': 4}]
