import torch
import torch.nn as nn


class ConvNet(nn.Module):
    """
    A network with a single convolution layer. This is used for testing flop
    count for convolution layers.
    """

    def __init__(self, conv_dim: 'int', input_dim: 'int', output_dim: 'int',
        kernel_size: 'int', spatial_dim: 'int', stride: 'int', padding:
        'int', groups_num: 'int') ->None:
        super(ConvNet, self).__init__()
        if conv_dim == 1:
            convLayer = nn.Conv1d
        elif conv_dim == 2:
            convLayer = nn.Conv2d
        else:
            convLayer = nn.Conv3d
        self.conv = convLayer(input_dim, output_dim, kernel_size, stride,
            padding, groups=groups_num)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.conv(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'conv_dim': 4, 'input_dim': 4, 'output_dim': 4,
        'kernel_size': 4, 'spatial_dim': 4, 'stride': 1, 'padding': 4,
        'groups_num': 1}]
