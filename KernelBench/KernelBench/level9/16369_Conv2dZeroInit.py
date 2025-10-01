import torch
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn


class Conv2dZeroInit(nn.Conv2d):

    def __init__(self, channels_in, channels_out, filter_size, stride=1,
        padding=0, logscale=3.0):
        super().__init__(channels_in, channels_out, filter_size, stride=
            stride, padding=padding)
        self.register_parameter('logs', nn.Parameter(torch.zeros(
            channels_out, 1, 1)))
        self.logscale_factor = logscale

    def reset_parameters(self):
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        out = super().forward(input)
        return out * torch.exp(self.logs * self.logscale_factor)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels_in': 4, 'channels_out': 4, 'filter_size': 4}]
