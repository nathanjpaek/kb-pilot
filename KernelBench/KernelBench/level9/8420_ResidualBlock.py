import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block from R2D3/IMPALA

    Taken from [1,2]
    """

    def __init__(self, num_channels, first_conv_weight_scale):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bias1 = nn.Parameter(torch.zeros([num_channels, 1, 1]))
        self.bias2 = nn.Parameter(torch.zeros([num_channels, 1, 1]))
        self.bias3 = nn.Parameter(torch.zeros([num_channels, 1, 1]))
        self.bias4 = nn.Parameter(torch.zeros([num_channels, 1, 1]))
        self.scale = nn.Parameter(torch.ones([num_channels, 1, 1]))
        with torch.no_grad():
            self.conv2.weight *= 0
            self.conv1.weight *= first_conv_weight_scale

    def forward(self, x):
        x = F.relu(x, inplace=True)
        original = x
        x = x + self.bias1
        x = self.conv1(x)
        x = x + self.bias2
        x = F.relu(x, inplace=True)
        x = x + self.bias3
        x = self.conv2(x)
        x = x * self.scale
        x = x + self.bias4
        return original + x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_channels': 4, 'first_conv_weight_scale': 1.0}]
