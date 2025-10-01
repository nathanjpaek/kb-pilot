import torch
from torch.nn import functional as F


class ConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels,
            kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x, in_weights=None):
        residual = x
        if in_weights is None:
            out = self.relu(self.in1(self.conv1(x)))
            out = self.in2(self.conv2(out))
        else:
            out = self.conv1(x)
            out = F.instance_norm(out, weight=in_weights['in1.weight'],
                bias=in_weights['in1.bias'])
            out = self.relu(out)
            out = self.conv2(out)
            out = F.instance_norm(out, weight=in_weights['in2.weight'],
                bias=in_weights['in2.bias'])
        out = out + residual
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]
