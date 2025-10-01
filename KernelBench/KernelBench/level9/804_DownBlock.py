import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, input_size, output_size, kernel_size=3, stride=1,
        padding=1, bias=True, norm=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride,
            padding, bias=bias)
        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)
        self.act = nn.PReLU()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)
        return self.act(out)


class DeconvBlock(nn.Module):

    def __init__(self, input_size, output_size, kernel_size=4, stride=2,
        padding=1, bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_size, output_size,
            kernel_size, stride, padding, bias=bias)
        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)
        self.activation = activation
        self.act = nn.PReLU()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)
        if self.activation is not None:
            return self.act(out)
        else:
            return out


class DownBlock(nn.Module):

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias
        =True, activation='prelu', norm=None):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size,
            stride, padding, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size,
            stride, padding, activation=activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size,
            stride, padding, norm=None)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_filter': 4}]
