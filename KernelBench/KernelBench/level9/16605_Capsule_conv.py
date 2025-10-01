import torch
import torch.nn as nn


def Squash(x):
    l2norm = x.norm(dim=-1, keepdim=True)
    unit_v = x / l2norm
    squashed_v = l2norm.pow(2) / (1 + l2norm.pow(2))
    x = unit_v * squashed_v
    return x


class Capsule_conv(nn.Module):

    def __init__(self, in_channels, out_channels, cap_dim):
        super(Capsule_conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cap_dim = cap_dim
        self.kernel_size = 9
        self.stride = 2
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=
            self.out_channels * self.cap_dim, kernel_size=self.kernel_size,
            stride=self.stride)

    def forward(self, x):
        """

        :param x: shape = 256 x 20 x 20. Output of convolution operation
        :return: output of primary capsules
        """
        x = self.conv(x)
        x = x.view(x.shape[0], -1, self.cap_dim)
        x = Squash(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'cap_dim': 4}]
