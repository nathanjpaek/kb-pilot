import torch
import torch.nn as nn
from torch.nn.init import calculate_gain
import torch.nn.parallel


class FilterNorm(nn.Module):

    def __init__(self, in_channels, kernel_size, filter_type, nonlinearity=
        'linear', running_std=False, running_mean=False):
        assert filter_type in ('spatial', 'channel')
        if filter_type == 'spatial':
            assert in_channels == 1
        else:
            assert in_channels >= 1
        super(FilterNorm, self).__init__()
        self.in_channels = in_channels
        self.filter_type = filter_type
        self.runing_std = running_std
        self.runing_mean = running_mean
        std = calculate_gain(nonlinearity) / kernel_size
        if running_std:
            self.std = nn.Parameter(torch.randn(in_channels * kernel_size **
                2) * std, requires_grad=True)
        else:
            self.std = std
        if running_mean:
            self.mean = nn.Parameter(torch.randn(in_channels * kernel_size **
                2), requires_grad=True)

    def forward(self, x):
        if self.filter_type == 'spatial':
            b, c, h, w = x.size()
            x = x - x.mean(dim=1).view(b, 1, h, w)
            x = x / (x.std(dim=1).view(b, 1, h, w) + 1e-10)
            if self.runing_std:
                x = x * self.std[None, :, None, None]
            else:
                x = x * self.std
            if self.runing_mean:
                x = x + self.mean[None, :, None, None]
        elif self.filter_type == 'channel':
            b = x.size(0)
            c = self.in_channels
            x = x.view(b, c, -1)
            x = x - x.mean(dim=2).view(b, c, 1)
            x = x / (x.std(dim=2).view(b, c, 1) + 1e-10)
            x = x.view(b, -1)
            if self.runing_std:
                x = x * self.std[None, :]
            else:
                x = x * self.std
            if self.runing_mean:
                x = x + self.mean[None, :]
        else:
            raise RuntimeError('Unsupported filter type {}'.format(self.
                filter_type))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 1, 'kernel_size': 4, 'filter_type': 'spatial'}]
