import torch
from torch import nn


class CircularPad(nn.Module):

    def __init__(self, pad):
        super(CircularPad, self).__init__()
        self.pad = pad
        self.zeropad = torch.nn.modules.padding.ConstantPad2d((pad, pad, 0,
            0), 0)

    def forward(self, x):
        x = torch.cat([x[..., -self.pad:, :], x, x[..., :self.pad, :]], dim=-2)
        x = self.zeropad(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'pad': 4}]
