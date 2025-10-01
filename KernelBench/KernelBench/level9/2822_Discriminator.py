import torch
import torch.nn as nn


class Down2d(nn.Module):
    """docstring for Down2d."""

    def __init__(self, in_channel, out_channel, kernel, stride, padding):
        super(Down2d, self).__init__()
        self.c1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel,
            stride=stride, padding=padding)
        self.n1 = nn.InstanceNorm2d(out_channel)
        self.c2 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel,
            stride=stride, padding=padding)
        self.n2 = nn.InstanceNorm2d(out_channel)

    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.n1(x1)
        x2 = self.c2(x)
        x2 = self.n2(x2)
        x3 = x1 * torch.sigmoid(x2)
        return x3


class Discriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(self):
        super(Discriminator, self).__init__()
        self.d1 = Down2d(5, 32, (3, 9), (1, 1), (1, 4))
        self.d2 = Down2d(36, 32, (3, 8), (1, 2), (1, 3))
        self.d3 = Down2d(36, 32, (3, 8), (1, 2), (1, 3))
        self.d4 = Down2d(36, 32, (3, 6), (1, 2), (1, 2))
        self.conv = nn.Conv2d(36, 1, (36, 5), (36, 1), (0, 2))
        self.pool = nn.AvgPool2d((1, 64))

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c1 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c1], dim=1)
        x = self.d1(x)
        c2 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c2], dim=1)
        x = self.d2(x)
        c3 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c3], dim=1)
        x = self.d3(x)
        c4 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c4], dim=1)
        x = self.d4(x)
        c5 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c5], dim=1)
        x = self.conv(x)
        x = self.pool(x)
        x = torch.squeeze(x)
        x = torch.tanh(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 512, 512]), torch.rand([4, 4, 1, 1])]


def get_init_inputs():
    return [[], {}]
