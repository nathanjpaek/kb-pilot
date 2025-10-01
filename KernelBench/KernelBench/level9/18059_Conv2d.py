import torch
from torch import nn


class Conv2d(nn.Module):
    """docstring for Conv2d

    Attributes
    ----------
    bn : TYPE
        Description
    conv : TYPE
        Description
    relu : TYPE
        Description
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        relu=True, same_padding=False, bn=False):
        """Summary

        Parameters
        ----------
        in_channels : TYPE
            Description
        out_channels : TYPE
            Description
        kernel_size : TYPE
            Description
        stride : int, optional
            Description
        relu : bool, optional
            Description
        same_padding : bool, optional
            Description
        bn : bool, optional
            Description
        """
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding=padding, bias=not bn)
        nn.init.xavier_normal_(self.conv.weight)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, affine=True
            ) if bn else None
        self.relu = nn.LeakyReLU(negative_slope=0.1) if relu else None

    def forward(self, x):
        """Summary

        Parameters
        ----------
        x : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
