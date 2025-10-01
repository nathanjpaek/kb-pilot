import torch
import torch.nn as nn
from torch.nn.modules.loss import *
from torch.nn.modules import *
from torch.optim import *
from torch.optim.lr_scheduler import *
import torch.distributed
import torch.backends


class ChannelSqueezeAndSpatialExcitation(nn.Module):
    """
    The sSE (Channel Squeeze and Spatial Excitation) block from the
    [Concurrent Spatial and Channel ‘Squeeze & Excitation’
    in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579) paper.

    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178

    Shape:

    - Input: (batch, channels, height, width)
    - Output: (batch, channels, height, width) (same shape as input)
    """

    def __init__(self, in_channels: 'int'):
        """
        Args:
            in_channels (int): The number of channels
                in the feature map of the input.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x: 'torch.Tensor'):
        """Forward call."""
        input_x = x
        x = self.conv(x)
        x = torch.sigmoid(x)
        x = torch.mul(input_x, x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
