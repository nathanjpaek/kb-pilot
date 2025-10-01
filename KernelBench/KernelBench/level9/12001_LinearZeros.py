import torch
import torch.nn as nn


class LinearZeros(nn.Module):

    def __init__(self, in_channels, out_channels, logscale_factor=3):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.logscale_factor = logscale_factor
        self.logs = nn.Parameter(torch.zeros(out_channels))

    def forward(self, input):
        output = self.linear(input)
        return output * torch.exp(self.logs * self.logscale_factor)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
