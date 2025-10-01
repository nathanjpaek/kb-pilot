import torch
import torch.nn as nn


class OutputTransition(nn.Module):

    def __init__(self, out_ch):
        super(OutputTransition, self).__init__()
        self.up_conv = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        out = self.up_conv(x)
        return out


def get_inputs():
    return [torch.rand([4, 64, 64, 64])]


def get_init_inputs():
    return [[], {'out_ch': 4}]
