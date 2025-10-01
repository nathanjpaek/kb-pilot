import torch
from torch import nn
import torch.utils.data


class ResidualBlock(nn.Module):

    def __init__(self, channels, reduction):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU(num_parameters=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.att_pool = nn.AdaptiveAvgPool2d(1)
        self.att_conv1 = nn.Conv2d(channels, channels // reduction,
            kernel_size=1, padding=0)
        self.att_prelu = nn.PReLU(num_parameters=channels // reduction)
        self.att_conv2 = nn.Conv2d(channels // reduction, channels,
            kernel_size=1, padding=0)
        self.att_sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        attenuation = self.att_pool(residual)
        attenuation = self.att_conv1(attenuation)
        attenuation = self.att_prelu(attenuation)
        attenuation = self.att_conv2(attenuation)
        attenuation = self.att_sigmoid(attenuation)
        return x + residual * attenuation


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4, 'reduction': 4}]
