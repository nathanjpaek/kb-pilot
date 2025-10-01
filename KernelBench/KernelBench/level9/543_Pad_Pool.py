import torch
from torch import nn


class Pad_Pool(nn.Module):
    """
    Implements a padding layer in front of pool1d layers used in our architectures to achieve padding=same output shape 
    Pads 0 to the left and 1 to the right side of x 
    """

    def __init__(self, left=0, right=1, value=0):
        super().__init__()
        self.left = left
        self.right = right
        self.value = value

    def forward(self, x):
        return nn.ConstantPad1d(padding=(self.left, self.right), value=self
            .value)(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
