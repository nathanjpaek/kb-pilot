import torch
from torch import nn


class GlobalAttention(nn.Module):

    def __init__(self, dims):
        super(GlobalAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(dims, dims, 1)

    def forward(self, x, y):
        att = torch.sigmoid(self.conv(self.pool(x + y)))
        return x * att + y * (1 - att)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dims': 4}]
