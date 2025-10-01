import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Vanilla convolutional residual block from seminal paper by He et al.

    Use of instance normalization suggested by Ulyanov et al. in
    https://arxiv.org/pdf/1607.08022.pdf%C2%A0%C2%A0%C2%A0%C2%A0.
    """

    def __init__(self, filters=128):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, (3, 3), padding=(1, 1),
            padding_mode='reflect')
        self.in_norm1 = nn.InstanceNorm2d(filters, affine=True)
        self.conv2 = nn.Conv2d(filters, filters, (3, 3), padding=(1, 1),
            padding_mode='reflect')
        self.in_norm2 = nn.InstanceNorm2d(filters, affine=True)

    def forward(self, x):
        a = self.conv1(x)
        b = self.in_norm1(a)
        c = F.relu(b)
        d = self.conv2(c)
        e = self.in_norm2(d)
        return F.relu(e + x)


def get_inputs():
    return [torch.rand([4, 128, 4, 4])]


def get_init_inputs():
    return [[], {}]
