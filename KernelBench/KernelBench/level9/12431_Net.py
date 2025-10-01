import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_size, out_size, kernel=3, stride=1, padding=1,
        activ='relu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel, stride, padding)
        self.norm = norm
        self.activ = activ
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(out_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(out_size)
        if self.activ == 'relu':
            self.act = nn.ReLU(True)
        elif self.activ == 'prelu':
            self.act = nn.PReLU()
        elif self.activ == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activ == 'tanh':
            self.act = nn.Tanh()

    def forward(self, x):
        if self.norm is not None:
            x = self.bn(self.conv(x))
        else:
            x = self.conv(x)
        if self.activ is not None:
            return self.act(x)
        else:
            return x


class DeconvBlock(nn.Module):

    def __init__(self, in_size, out_size, kernel=4, stride=2, padding=1,
        activ='relu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_size, out_size, kernel, stride,
            padding)
        self.norm = norm
        self.activ = activ
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(out_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(out_size)
        if self.activ == 'relu':
            self.act = nn.ReLU(True)
        elif self.activ == 'prelu':
            self.act = nn.PReLU()
        elif self.activ == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activ == 'tanh':
            self.act = nn.Tanh()

    def forward(self, x):
        if self.norm is not None:
            x = self.bn(self.deconv(x))
        else:
            x = self.deconv(x)
        if self.activ is not None:
            return self.act(x)
        else:
            return x


class DownBlock(torch.nn.Module):

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2,
        activation='prelu'):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size,
            stride, padding, activation)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size,
            stride, padding, activation)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size,
            stride, padding, activation)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class UpBlock(torch.nn.Module):

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2,
        activation='prelu'):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size,
            stride, padding, activation)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size,
            stride, padding, activation)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size,
            stride, padding, activation)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class Net(nn.Module):

    def __init__(self, channels, filters, features, scale_fact):
        super(Net, self).__init__()
        self.lay1 = ConvBlock(in_size=channels, out_size=features, kernel=3,
            stride=1, padding=1, activ='prelu')
        self.lay2 = ConvBlock(in_size=features, out_size=filters, kernel=1,
            stride=1, padding=0, activ='prelu')
        self.down1 = DownBlock(num_filter=filters, kernel_size=6, stride=2,
            padding=2, activation='prelu')
        self.up1 = UpBlock(num_filter=filters, kernel_size=6, stride=2,
            padding=2, activation='prelu')
        self.down2 = DownBlock(num_filter=filters, kernel_size=6, stride=2,
            padding=2, activation='prelu')
        self.up2 = UpBlock(num_filter=filters, kernel_size=6, stride=2,
            padding=2, activation='prelu')
        self.out1 = DeconvBlock(in_size=filters, out_size=features, kernel=
            1, stride=1, padding=0, activ='prelu')
        self.out2 = DeconvBlock(in_size=features, out_size=channels, kernel
            =2, stride=2, padding=0, activ='prelu')

    def forward(self, x):
        x = self.lay1(x)
        x = self.lay2(x)
        x = self.down1(x)
        x = self.up1(x)
        x = self.down2(x)
        x = self.up2(x)
        x = self.out1(x)
        x = self.out2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4, 'filters': 4, 'features': 4, 'scale_fact': 1.0}
        ]
