import torch
from torch import nn


class _SepConv1d(nn.Module):
    """A simple separable convolution implementation.

    The separable convlution is a method to reduce number of the parameters
    in the deep learning network for slight decrease in predictions quality.
    """

    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()
        self.depthwise = nn.Conv1d(ni, ni, kernel, stride, padding=pad,
            groups=ni)
        self.pointwise = nn.Conv1d(ni, no, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'ni': 4, 'no': 4, 'kernel': 4, 'stride': 1, 'pad': 4}]
