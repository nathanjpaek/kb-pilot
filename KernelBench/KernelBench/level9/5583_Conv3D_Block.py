import torch
import torch.nn as nn


def define_norm(n_channel, norm_type, n_group=None, dim_mode=2):
    if norm_type == 'bn':
        if dim_mode == 2:
            return nn.BatchNorm2d(n_channel)
        elif dim_mode == 3:
            return nn.BatchNorm3d(n_channel)
    elif norm_type == 'gn':
        if n_group is None:
            n_group = 2
        return nn.GroupNorm(n_group, n_channel)
    elif norm_type == 'in':
        return nn.GroupNorm(n_channel, n_channel)
    elif norm_type == 'ln':
        return nn.GroupNorm(1, n_channel)
    elif norm_type is None:
        return
    else:
        return ValueError('Normalization type - ' + norm_type +
            ' is not defined yet')


class Conv3D_Block(nn.Module):
    """ 
  use conv3D than multiple Conv2D blocks (for a sake of reducing computational burden)
  INPUT dimension: BxCxTxHxW
  """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=1, norm_type=None):
        super(Conv3D_Block, self).__init__()
        self.norm_type = norm_type
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1,
            kernel_size, kernel_size), stride=(1, stride, stride), padding=
            (1, padding, padding))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.norm_layer = define_norm(out_channels, norm_type, dim_mode=3)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
