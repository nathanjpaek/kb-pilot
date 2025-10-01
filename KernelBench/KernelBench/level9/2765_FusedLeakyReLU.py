import torch
from torch import nn
from torch.nn import functional as F


class FusedLeakyReLU(nn.Module):

    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        bias = self.bias[None, :, None, None]
        try:
            out = F.leaky_relu(input + bias, negative_slope=self.negative_slope
                ) * self.scale
        except Exception:
            code.interact('Something is wrong with bias', local={**globals(
                ), **locals()})
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channel': 4}]
