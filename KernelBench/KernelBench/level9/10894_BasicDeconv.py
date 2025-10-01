import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicDeconv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        use_bn=False):
        super(BasicDeconv, self).__init__()
        self.use_bn = use_bn
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels,
            kernel_size, stride=stride, bias=not self.use_bn)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True
            ) if self.use_bn else None

    def forward(self, x):
        x = self.tconv(x)
        if self.use_bn:
            x = self.bn(x)
        return F.relu(x, inplace=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
