import torch
from torch import nn


class Pad_Pool2d(nn.Module):
    """
    Implements a padding layer in front of pool1d layers used in our architectures to achieve padding=same output shape 
    Pads 0 to the left and 1 to the right side of x 
    """

    def __init__(self, left=0, right=1, top=0, bottom=1, value=0):
        super().__init__()
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.value = value

    def forward(self, x):
        return nn.ConstantPad2d(padding=(self.left, self.right, self.top,
            self.bottom), value=self.value)(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
