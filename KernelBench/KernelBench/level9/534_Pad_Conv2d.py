import math
import torch
from torch import nn


class Pad_Conv2d(nn.Module):
    """
    Implements a padding layer in front of conv2d layers used in our architectures to achieve padding=same output shape 
    Pads 0 to the left and 1 to the right side of x 
    Input:
    kernel as a tuple (kx, ky)
    Output:
    Padded tensor for the following convolution s.t. padding=same
    """

    def __init__(self, kernel, value=0):
        super().__init__()
        kernel_x, kernel_y = kernel
        self.value = value
        self.left = max(math.floor(kernel_y / 2) - 1, 0)
        self.right = max(math.floor(kernel_y / 2), 0)
        self.top = max(math.floor(kernel_x / 2) - 1, 0)
        self.bottom = max(math.floor(kernel_x / 2), 0)

    def forward(self, x):
        return nn.ConstantPad2d((self.left, self.right, self.top, self.
            bottom), self.value)(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel': [4, 4]}]
