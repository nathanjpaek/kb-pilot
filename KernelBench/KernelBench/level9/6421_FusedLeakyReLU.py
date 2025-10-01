import torch
from torch import nn
from torch.nn import functional as F


class FusedLeakyReLU(nn.Module):

    def __init__(self, channel):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.scale = 1.414

    def forward(self, input):
        shape = 1, self.bias.shape[0], 1, 1
        return self.scale * F.leaky_relu(input + self.bias.view(shape),
            negative_slope=0.2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channel': 4}]
