import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def bilinear_kernel(size, normalize=False):
    """
    Make a 2D bilinear kernel suitable for upsampling/downsampling with
    normalize=False/True. The kernel is size x size square.

    Take
        size: kernel size (square)
        normalize: whether kernel sums to 1 (True) or not

    Give
        kernel: np.array with bilinear kernel coefficient
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    kernel = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) /
        factor)
    if normalize:
        kernel /= kernel.sum()
    return kernel


class Interpolator(nn.Module):
    """
    Interpolate by de/up/backward convolution with a bilinear kernel.

    Take
        channel_dim: the input channel dimension
        rate: upsampling rate, that is 4 -> 4x upsampling
        odd: the kernel parity, which is too much to explain here for now, but
             will be handled automagically in the future, promise.
        normalize: whether kernel sums to 1
    """

    def __init__(self, channel_dim, rate, odd=True, normalize=False):
        super().__init__()
        self.rate = rate
        ksize = rate * 2
        if odd:
            ksize -= 1
        kernel = torch.from_numpy(bilinear_kernel(ksize, normalize))
        weight = torch.zeros(channel_dim, channel_dim, ksize, ksize)
        for k in range(channel_dim):
            weight[k, k] = kernel
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        return F.conv_transpose2d(x, self.weight, stride=self.rate)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channel_dim': 4, 'rate': 4}]
