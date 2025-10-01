import torch
import torch.nn as nn


class GapAggregator(nn.Module):

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.pool(x).squeeze(3).squeeze(2)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
