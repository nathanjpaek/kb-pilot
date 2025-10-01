import torch
import numpy as np
from torch import nn


class SinActivation(nn.Module):

    def __init__(self):
        super(SinActivation, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class ConLinear(nn.Module):

    def __init__(self, ch_in, ch_out, is_first=False, bias=True):
        super(ConLinear, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, bias
            =bias)
        if is_first:
            nn.init.uniform_(self.conv.weight, -np.sqrt(9 / ch_in), np.sqrt
                (9 / ch_in))
        else:
            nn.init.uniform_(self.conv.weight, -np.sqrt(3 / ch_in), np.sqrt
                (3 / ch_in))

    def forward(self, x):
        return self.conv(x)


class LFF(nn.Module):

    def __init__(self, hidden_size):
        super(LFF, self).__init__()
        self.ffm = ConLinear(2, hidden_size, is_first=True)
        self.activation = SinActivation()

    def forward(self, x):
        x = self.ffm(x)
        x = self.activation(x)
        return x


def get_inputs():
    return [torch.rand([4, 2, 64, 64])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
