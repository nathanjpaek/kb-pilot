import torch
import torch.nn as nn


class ChannelNorm(nn.Module):

    def __init__(self):
        super(ChannelNorm, self).__init__()

    def forward(self, x):
        divider = torch.max(torch.max(torch.abs(x), dim=0)[0], dim=1)[0
            ] + 1e-05
        divider = divider.unsqueeze(0).unsqueeze(2)
        divider = divider.repeat(x.size(0), 1, x.size(2))
        x = x / divider
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
