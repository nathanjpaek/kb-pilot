import torch
import torch.utils.data
import torch.nn as nn


class MaxPoolBranch(nn.Module):
    """
    PolyNet specific max pooling branch block.
    """

    def __init__(self):
        super(MaxPoolBranch, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = self.pool(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
