import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn


class BlockWidth2d(nn.Module):

    def __init__(self, width) ->None:
        super().__init__()
        self.conv = nn.Conv2d(width, width, kernel_size=3, padding=1)

    def forward(self, x):
        x = x + F.leaky_relu(self.conv(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'width': 4}]
