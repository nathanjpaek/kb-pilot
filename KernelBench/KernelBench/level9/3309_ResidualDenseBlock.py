import torch
from torch import Tensor
import torch.nn as nn


class ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.

    Args:
            channels (int): The number of channels in the input image.
            growths (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: 'int', growths: 'int') ->None:
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels + growths * 0, growths, (3, 3), (1,
            1), (1, 1))
        self.conv2 = nn.Conv2d(channels + growths * 1, growths, (3, 3), (1,
            1), (1, 1))
        self.conv3 = nn.Conv2d(channels + growths * 2, growths, (3, 3), (1,
            1), (1, 1))
        self.conv4 = nn.Conv2d(channels + growths * 3, growths, (3, 3), (1,
            1), (1, 1))
        self.conv5 = nn.Conv2d(channels + growths * 4, channels, (3, 3), (1,
            1), (1, 1))
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: 'Tensor') ->Tensor:
        identity = x
        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3,
            out4], 1)))
        out = out5 * 0.2 + identity
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4, 'growths': 4}]
