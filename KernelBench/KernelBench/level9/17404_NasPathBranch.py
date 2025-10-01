import torch
import torch.utils.data
import torch.nn as nn


def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False):
    """
    Convolution 1x1 layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
        kernel_size=1, stride=stride, groups=groups, bias=bias)


def nasnet_avgpool1x1_s2():
    """
    NASNet specific 1x1 Average pooling layer with stride 2.
    """
    return nn.AvgPool2d(kernel_size=1, stride=2, count_include_pad=False)


class NasPathBranch(nn.Module):
    """
    NASNet specific `path` branch (auxiliary block).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    extra_padding : bool, default False
        Whether to use extra padding.
    """

    def __init__(self, in_channels, out_channels, extra_padding=False):
        super(NasPathBranch, self).__init__()
        self.extra_padding = extra_padding
        self.avgpool = nasnet_avgpool1x1_s2()
        self.conv = conv1x1(in_channels=in_channels, out_channels=out_channels)
        if self.extra_padding:
            self.pad = nn.ZeroPad2d(padding=(0, 1, 0, 1))

    def forward(self, x):
        if self.extra_padding:
            x = self.pad(x)
            x = x[:, :, 1:, 1:].contiguous()
        x = self.avgpool(x)
        x = self.conv(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
