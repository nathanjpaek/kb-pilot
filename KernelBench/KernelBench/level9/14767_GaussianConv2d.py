import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
from torch.nn.parameter import Parameter


class GaussianConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, ksize=5):
        """Applies 2-D Gaussian Blur.

    Args:
      in_channels: An integer indicates input channel dimension.
      out_channels: An integer indicates output channel dimension.
      ksize: An integer indicates Gaussian kernel size.
    """
        super(GaussianConv2d, self).__init__()
        weight = (np.arange(ksize, dtype=np.float32) - ksize // 2) ** 2
        weight = np.sqrt(weight[None, :] + weight[:, None])
        weight = np.reshape(weight, (1, 1, ksize, ksize)) / weight.sum()
        self.weight = Parameter(torch.Tensor(weight).expand(out_channels, -
            1, -1, -1))
        self._in_channels = in_channels
        self._out_channels = out_channels

    def forward(self, x):
        with torch.no_grad():
            return F.conv2d(x, self.weight, groups=self._in_channels)


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
