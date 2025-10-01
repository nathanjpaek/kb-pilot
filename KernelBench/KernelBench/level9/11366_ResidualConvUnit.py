import torch
import torch.fft
import torch.nn as nn
import torch.utils.cpp_extension


class ResidualConvUnit(nn.Module):

    def __init__(self, cin, activation, bn):
        super().__init__()
        self.conv = nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1,
            bias=True)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.skip_add.add(self.conv(x), x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'cin': 4, 'activation': 4, 'bn': 4}]
