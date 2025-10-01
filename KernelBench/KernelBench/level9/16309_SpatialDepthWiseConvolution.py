from torch.nn import Module
import math
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional
import torch.autograd


class SpatialDepthWiseConvolution(Module):
    """
    ## Spatial Depth Wise Convolution

    This is actually slower
    """

    def __init__(self, d_k: 'int', kernel_size: 'int'=3):
        """
        * `d_k` is the number of channels in each head
        """
        super().__init__()
        self.kernel_size = kernel_size
        rng = 1 / math.sqrt(kernel_size)
        self.kernels = nn.Parameter(torch.zeros((kernel_size, d_k)).
            uniform_(-rng, rng))

    def forward(self, x: 'torch.Tensor'):
        """
        `x` has shape `[seq_len, batch_size, heads, d_k]`
        """
        res = x * self.kernels[0].view(1, 1, 1, -1)
        for i in range(1, len(self.kernels)):
            res[i:] += x[:-i] * self.kernels[i].view(1, 1, 1, -1)
        return res


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_k': 4}]
