import math
import torch
from torch import nn
import torch.nn.functional as F


class ScaledLeakyReLUSin(nn.Module):

    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input):
        out_lr = F.leaky_relu(input[:, ::2], negative_slope=self.negative_slope
            )
        out_sin = torch.sin(input[:, 1::2])
        out = torch.cat([out_lr, out_sin], 1)
        return out * math.sqrt(2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
