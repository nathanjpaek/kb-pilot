import math
import torch
from torch import nn


class Pad_Conv(nn.Module):
    """
    Implements a padding layer in front of conv1d layers used in our architectures to achieve padding=same output shape 
    Pads 0 to the left and 1 to the right side of x 
    """

    def __init__(self, kernel_size, value=0):
        super().__init__()
        self.value = value
        self.left = max(math.floor(kernel_size / 2) - 1, 0)
        self.right = max(math.floor(kernel_size / 2), 0)

    def forward(self, x):
        return nn.ConstantPad1d(padding=(self.left, self.right), value=self
            .value)(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4}]
