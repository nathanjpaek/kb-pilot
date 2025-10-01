import torch
import torch.nn as nn


class EQConv2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding)
        self.scale = (gain / kernel_size ** 2 * in_channels) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.
            shape[0], 1, 1)


class PixelNorm(nn.Module):

    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-08

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) +
            self.epsilon)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super(ConvBlock, self).__init__()
        self.use_pn = use_pixelnorm
        self.conv1 = EQConv2D(in_channels, out_channels)
        self.conv2 = EQConv2D(out_channels, out_channels)
        self.LRelu = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()

    def forward(self, x):
        x = self.LRelu(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.LRelu(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
