import torch
import torch.nn as nn


class MaxPoolBranch(nn.Module):
    """
    InceptionV4 specific max pooling branch block.
    """

    def __init__(self, kernel_size=3, stride=2, padding=0):
        super(MaxPoolBranch, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
            padding=padding)

    def forward(self, x):
        x = self.pool(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
