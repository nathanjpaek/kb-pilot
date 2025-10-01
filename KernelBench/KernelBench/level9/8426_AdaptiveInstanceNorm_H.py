import torch
from torch import nn
import torch.utils.data
import torch.utils.data.distributed


class AdaptiveInstanceNorm_H(nn.Module):

    def __init__(self, in_channel, map_size):
        super().__init__()
        self.norm = nn.LayerNorm([map_size, map_size])
        self.weight = nn.Parameter(1000.0 + torch.randn(1, in_channel, 1, 1))
        self.beta = nn.Parameter(0.0 + torch.randn(1, in_channel, 1, 1))

    def forward(self, input, style=0):
        out = self.norm(input)
        out = 0.01 * out + out.detach() * self.weight + self.beta
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'map_size': 4}]
