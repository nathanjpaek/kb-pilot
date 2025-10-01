import torch
import torch.nn as nn
from torchvision.transforms import *


class ConvBlock(nn.Module):

    def __init__(self, input_size, output_size, kernel_size=3, stride=1,
        padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride,
            padding, bias=bias)
        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)
        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.1, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)
        if self.activation is not None:
            return self.act(out)
        else:
            return out


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
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.1, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)
        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ResnetBlock(nn.Module):

    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias
        =True, activation='prelu', norm='batch'):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filter, num_filter, kernel_size, stride,
            padding, bias=bias)
        self.conv2 = nn.Conv2d(num_filter, num_filter, kernel_size, stride,
            padding, bias=bias)
        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(num_filter)
        elif norm == 'instance':
            self.bn = nn.InstanceNorm2d(num_filter)
        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.1, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)
        if self.activation is not None:
            out = self.act(out)
        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)
        out = torch.add(out, residual)
        if self.activation is not None:
            out = self.act(out)
        return out


class PyramidModule(nn.Module):

    def __init__(self, num_inchannels, activation='prelu'):
        super(PyramidModule, self).__init__()
        self.l1_1 = ResnetBlock(num_inchannels, kernel_size=3, stride=1,
            padding=1, bias=True, activation=activation, norm=None)
        self.l1_2 = ResnetBlock(num_inchannels, kernel_size=3, stride=1,
            padding=1, bias=True, activation=activation, norm=None)
        self.l1_3 = ResnetBlock(num_inchannels, kernel_size=3, stride=1,
            padding=1, bias=True, activation=activation, norm=None)
        self.l1_4 = ResnetBlock(num_inchannels, kernel_size=3, stride=1,
            padding=1, bias=True, activation=activation, norm=None)
        self.l1_5 = ResnetBlock(num_inchannels, kernel_size=3, stride=1,
            padding=1, bias=True, activation=activation, norm=None)
        self.l2_1 = ResnetBlock(num_inchannels * 2, kernel_size=3, stride=1,
            padding=1, bias=True, activation=activation, norm=None)
        self.l2_2 = ResnetBlock(num_inchannels * 2, kernel_size=3, stride=1,
            padding=1, bias=True, activation=activation, norm=None)
        self.l2_3 = ResnetBlock(num_inchannels * 2, kernel_size=3, stride=1,
            padding=1, bias=True, activation=activation, norm=None)
        self.l2_4 = ResnetBlock(num_inchannels * 2, kernel_size=3, stride=1,
            padding=1, bias=True, activation=activation, norm=None)
        self.l3_1 = ResnetBlock(num_inchannels * 4, kernel_size=3, stride=1,
            padding=1, bias=True, activation=activation, norm=None)
        self.l3_2 = ResnetBlock(num_inchannels * 4, kernel_size=3, stride=1,
            padding=1, bias=True, activation=activation, norm=None)
        self.l3_3 = ResnetBlock(num_inchannels * 4, kernel_size=3, stride=1,
            padding=1, bias=True, activation=activation, norm=None)
        self.down1 = ConvBlock(num_inchannels, num_inchannels * 2, 4, 2, 1,
            bias=True, activation=activation, norm=None)
        self.down2 = ConvBlock(num_inchannels * 2, num_inchannels * 4, 4, 2,
            1, bias=True, activation=activation, norm=None)
        self.up1 = DeconvBlock(num_inchannels * 2, num_inchannels, 4, 2, 1,
            bias=True, activation=activation, norm=None)
        self.up2 = DeconvBlock(num_inchannels * 4, num_inchannels * 2, 4, 2,
            1, bias=True, activation=activation, norm=None)
        self.final = ConvBlock(num_inchannels, num_inchannels, 3, 1, 1,
            bias=True, activation=activation, norm=None)

    def forward(self, x):
        out1_1 = self.l1_1(x)
        out2_1 = self.l2_1(self.down1(out1_1))
        out3_1 = self.l3_1(self.down2(out2_1))
        out1_2 = self.l1_2(out1_1 + self.up1(out2_1))
        out2_2 = self.l2_2(out2_1 + self.down1(out1_2) + self.up2(out3_1))
        out3_2 = self.l3_2(out3_1 + self.down2(out2_2))
        out1_3 = self.l1_3(out1_2 + self.up1(out2_2))
        out2_3 = self.l2_3(out2_2 + self.down1(out1_3) + self.up2(out3_2))
        out3_3 = self.l3_3(out3_2 + self.down2(out2_3))
        out1_4 = self.l1_4(out1_3 + self.up1(out2_3))
        out2_4 = self.l2_4(out2_3 + self.down1(out1_4) + self.up2(out3_3))
        out1_5 = self.l1_5(out1_4 + self.up1(out2_4))
        final = self.final(out1_5)
        return final


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_inchannels': 4}]
