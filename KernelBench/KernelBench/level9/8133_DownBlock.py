import torch
from torch import nn


class ConvBlock(torch.nn.Module):

    def __init__(self, input_size, output_size, kernel_size, stride,
        padding, bias=True):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size,
            stride, padding, bias=bias)
        self.act = torch.nn.PReLU()
        self.bn = nn.InstanceNorm2d(output_size)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return self.act(out)


class DeconvBlock(torch.nn.Module):

    def __init__(self, input_size, output_size, kernel_size, stride,
        padding, bias=True):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size,
            kernel_size, stride, padding, bias=bias)
        self.act = torch.nn.LeakyReLU()

    def forward(self, x):
        out = self.deconv(x)
        return self.act(out)


class DownBlock(torch.nn.Module):

    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(DownBlock, self).__init__()
        self.conv1 = ConvBlock(input_size, output_size, kernel_size, stride,
            padding, bias=False)
        self.conv2 = DeconvBlock(output_size, output_size, kernel_size,
            stride, padding, bias=False)
        self.conv3 = ConvBlock(output_size, output_size, kernel_size,
            stride, padding, bias=False)
        self.local_weight1 = nn.Conv2d(input_size, 2 * output_size,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.local_weight2 = nn.Conv2d(output_size, 2 * output_size,
            kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        lr = self.conv1(x)
        hr = self.conv2(lr)
        mean, var = self.local_weight1(x).chunk(2, dim=1)
        residue = mean + hr * (1 + var)
        l_residue = self.conv3(residue)
        mean, var = self.local_weight2(lr).chunk(2, dim=1)
        return mean + l_residue * (1 + var)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4, 'kernel_size': 4,
        'stride': 1, 'padding': 4}]
