import torch
import torch.nn as nn
import torch.nn.functional as F


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


class HSwish(nn.Module):
    """
    H-Swish activation function from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    inplace : bool
        Whether to use inplace version of the module.
    """

    def __init__(self, inplace=False):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class MobileNetV3Classifier(nn.Module):
    """
    MobileNetV3 classifier.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of middle channels.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    """

    def __init__(self, in_channels, out_channels, mid_channels, dropout_rate):
        super(MobileNetV3Classifier, self).__init__()
        self.use_dropout = dropout_rate != 0.0
        self.conv1 = conv1x1(in_channels=in_channels, out_channels=mid_channels
            )
        self.activ = HSwish(inplace=True)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = conv1x1(in_channels=mid_channels, out_channels=
            out_channels, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activ(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.conv2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'mid_channels': 4,
        'dropout_rate': 0.5}]
