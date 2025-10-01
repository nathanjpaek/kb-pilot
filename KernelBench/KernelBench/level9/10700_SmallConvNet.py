import torch
from typing import Tuple
import torch.nn as nn
from numpy import prod


class SmallConvNet(nn.Module):
    """
    A network with three conv layers. This is used for testing convolution
    layers for activation count.
    """

    def __init__(self, input_dim: 'int') ->None:
        super(SmallConvNet, self).__init__()
        conv_dim1 = 8
        conv_dim2 = 4
        conv_dim3 = 2
        self.conv1 = nn.Conv2d(input_dim, conv_dim1, 1, 1)
        self.conv2 = nn.Conv2d(conv_dim1, conv_dim2, 1, 2)
        self.conv3 = nn.Conv2d(conv_dim2, conv_dim3, 1, 2)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def get_gt_activation(self, x: 'torch.Tensor') ->Tuple[int, int, int]:
        x = self.conv1(x)
        count1 = prod(list(x.size()))
        x = self.conv2(x)
        count2 = prod(list(x.size()))
        x = self.conv3(x)
        count3 = prod(list(x.size()))
        return count1, count2, count3


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
