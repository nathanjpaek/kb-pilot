import torch
import torch.nn as nn


class ChannelRate(nn.Module):
    rates: 'torch.Tensor'

    def __init__(self, num_channels: 'int', device=None, dtype=None):
        super().__init__()
        kw = {'device': device, 'dtype': dtype}
        self.rates = nn.Parameter(torch.ones(num_channels, **kw))

    def forward(self, x):
        return x / self.rates.reshape(-1, 1, 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_channels': 4}]
