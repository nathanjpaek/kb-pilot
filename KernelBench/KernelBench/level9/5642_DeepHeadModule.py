import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from math import sqrt as sqrt
from itertools import product as product
import torchvision.transforms.functional as F
from torch.nn import functional as F


class DeepHeadModule(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(DeepHeadModule, self).__init__()
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._mid_channels = min(self._input_channels, 256)
        self.conv1 = nn.Conv2d(self._input_channels, self._mid_channels,
            kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self._mid_channels, self._mid_channels,
            kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self._mid_channels, self._mid_channels,
            kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv4 = nn.Conv2d(self._mid_channels, self._output_channels,
            kernel_size=1, dilation=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv4(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.
            conv1(x), inplace=True)), inplace=True)), inplace=True))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'output_channels': 4}]
