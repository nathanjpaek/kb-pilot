import torch
from torch import nn
import torch.backends.cudnn


def conv3x3(num_in, num_out):
    """Creates a 3x3 convolution building block module.

    Args:
      num_in: number of input feature maps
      num_out: number of output feature maps

    Returns:
      The 3x3 convolution module.
    """
    return nn.Conv2d(num_in, num_out, kernel_size=3, padding=1, bias=False)


class Conv3BN(nn.Module):

    def __init__(self, in_: 'int', out: 'int', bn=False):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


class UNetModule(nn.Module):

    def __init__(self, in_: 'int', out: 'int'):
        super().__init__()
        self.l1 = Conv3BN(in_, out)
        self.l2 = Conv3BN(out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_': 4, 'out': 4}]
