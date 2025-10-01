import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn


class BlockWidth1d(nn.Module):

    def __init__(self, width) ->None:
        super().__init__()
        self.conv = nn.Conv1d(width, width, kernel_size=5, padding=2)

    def forward(self, x):
        x = x + F.leaky_relu(self.conv(x))
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'width': 4}]
