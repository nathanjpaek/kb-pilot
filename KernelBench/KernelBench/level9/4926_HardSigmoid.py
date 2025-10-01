import torch
from torch import nn
import torch.nn.functional as F


class HardSigmoid(nn.Module):

    def __init__(self, slope=0.2, offset=0.5):
        super().__init__()
        self.slope = slope
        self.offset = offset

    def forward(self, x):
        x = self.slope * x + self.offset
        x = F.threshold(-x, -1, -1)
        x = F.threshold(-x, 0, 0)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
