import torch
import torch.nn as nn


class cSE(nn.Module):

    def __init__(self, out_channels):
        super().__init__()
        self.linear1 = nn.Linear(in_features=out_channels, out_features=int
            (out_channels / 2), bias=False)
        self.linear2 = nn.Linear(in_features=int(out_channels / 2),
            out_features=out_channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = nn.AdaptiveAvgPool2d(1)(x).view(b, c)
        y = self.linear1(y)
        y = torch.relu(y)
        y = self.linear2(y)
        y = torch.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'out_channels': 4}]
