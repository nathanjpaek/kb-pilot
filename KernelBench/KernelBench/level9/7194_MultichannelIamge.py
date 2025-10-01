import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModulatedConv2d(nn.Module):

    def __init__(self, channels_in, channels_out, style_dim, kernel_size,
        demodulate=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(channels_out, channels_in,
            kernel_size, kernel_size))
        self.modulation = nn.Linear(style_dim, channels_in, bias=True)
        self.modulation.bias.data.fill_(1.0)
        self.demodulate = demodulate
        if self.demodulate:
            self.register_buffer('style_inv', torch.randn(1, 1, channels_in,
                1, 1))
        self.scale = 1.0 / math.sqrt(channels_in * kernel_size ** 2)
        self.padding = kernel_size // 2

    def forward(self, x, style):
        modulation = self.get_modulation(style)
        x = modulation * x
        x = F.conv2d(x, self.weight, padding=self.padding)
        if self.demodulate:
            demodulation = self.get_demodulation(style)
            x = demodulation * x
        return x

    def get_modulation(self, style):
        style = self.modulation(style).view(style.size(0), -1, 1, 1)
        modulation = self.scale * style
        return modulation

    def get_demodulation(self, style):
        w = self.weight.unsqueeze(0)
        norm = torch.rsqrt((self.scale * self.style_inv * w).pow(2).sum([2,
            3, 4]) + 1e-08)
        demodulation = norm
        return demodulation.view(*demodulation.size(), 1, 1)


class MultichannelIamge(nn.Module):

    def __init__(self, channels_in, channels_out, style_dim, kernel_size=1):
        super().__init__()
        self.conv = ModulatedConv2d(channels_in, channels_out, style_dim,
            kernel_size, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, channels_out, 1, 1))

    def forward(self, hidden, style):
        out = self.conv(hidden, style)
        out = out + self.bias
        return out


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'channels_in': 4, 'channels_out': 4, 'style_dim': 4}]
