import copy
import torch
from torch import nn
import torch.utils.data


def get_activation(activation):
    if isinstance(activation, str):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky':
            return nn.LeakyReLU(negative_slope=0.1)
        elif activation == 'prelu':
            return nn.PReLU(num_parameters=1)
        elif activation == 'rrelu':
            return nn.RReLU()
        elif activation == 'silu':
            return nn.SiLU()
        elif activation == 'lin':
            return nn.Identity()
    else:
        return copy.deepcopy(activation)


def get_conv(dim=3):
    """Chooses an implementation for a convolution layer."""
    if dim == 3:
        return nn.Conv3d
    elif dim == 2:
        return nn.Conv2d
    else:
        raise ValueError('dim has to be 2 or 3')


def get_normalization(normtype: 'str', num_channels: 'int', dim: 'int'=3):
    """Chooses an implementation for a batch normalization layer."""
    if normtype is None or normtype == 'none':
        return nn.Identity()
    elif normtype.startswith('group'):
        if normtype == 'group':
            num_groups = 8
        elif len(normtype) > len('group') and normtype[len('group'):].isdigit(
            ):
            num_groups = int(normtype[len('group'):])
        else:
            raise ValueError(
                f'normtype "{normtype}" not understood. It should be "group<G>", where <G> is the number of groups.'
                )
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif normtype == 'instance':
        if dim == 3:
            return nn.InstanceNorm3d(num_channels)
        elif dim == 2:
            return nn.InstanceNorm2d(num_channels)
        else:
            raise ValueError('dim has to be 2 or 3')
    elif normtype == 'batch':
        if dim == 3:
            return nn.BatchNorm3d(num_channels)
        elif dim == 2:
            return nn.BatchNorm2d(num_channels)
        else:
            raise ValueError('dim has to be 2 or 3')
    else:
        raise ValueError(
            f"""Unknown normalization type "{normtype}".
Valid choices are "batch", "instance", "group" or "group<G>",where <G> is the number of groups."""
            )


def planar_kernel(x):
    """Returns a "planar" kernel shape (e.g. for 2D convolution in 3D space)
    that doesn't consider the first spatial dim (D)."""
    if isinstance(x, int):
        return 1, x, x
    else:
        return x


def planar_pad(x):
    """Returns a "planar" padding shape that doesn't pad along the first spatial dim (D)."""
    if isinstance(x, int):
        return 0, x, x
    else:
        return x


def get_maxpool(dim=3):
    """Chooses an implementation for a max-pooling layer."""
    if dim == 3:
        return nn.MaxPool3d
    elif dim == 2:
        return nn.MaxPool2d
    else:
        raise ValueError('dim has to be 2 or 3')


def conv3(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
    bias=True, planar=False, dim=3):
    """Returns an appropriate spatial convolution layer, depending on args.
    - dim=2: Conv2d with 3x3 kernel
    - dim=3 and planar=False: Conv3d with 3x3x3 kernel
    - dim=3 and planar=True: Conv3d with 1x3x3 kernel
    """
    if planar:
        stride = planar_kernel(stride)
        padding = planar_pad(padding)
        kernel_size = planar_kernel(kernel_size)
    return get_conv(dim)(in_channels, out_channels, kernel_size=kernel_size,
        stride=stride, padding=padding, bias=bias)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True, planar=
        False, activation='relu', normalization=None, full_norm=True, dim=3,
        conv_mode='same'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.normalization = normalization
        self.dim = dim
        padding = 1 if 'same' in conv_mode else 0
        self.conv1 = conv3(self.in_channels, self.out_channels, planar=
            planar, dim=dim, padding=padding)
        self.conv2 = conv3(self.out_channels, self.out_channels, planar=
            planar, dim=dim, padding=padding)
        if self.pooling:
            kernel_size = 2
            if planar:
                kernel_size = planar_kernel(kernel_size)
            self.pool = get_maxpool(dim)(kernel_size=kernel_size, ceil_mode
                =True)
            self.pool_ks = kernel_size
        else:
            self.pool = nn.Identity()
            self.pool_ks = -123
        self.act1 = get_activation(activation)
        self.act2 = get_activation(activation)
        if full_norm:
            self.norm0 = get_normalization(normalization, self.out_channels,
                dim=dim)
        else:
            self.norm0 = nn.Identity()
        self.norm1 = get_normalization(normalization, self.out_channels,
            dim=dim)

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm0(y)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.norm1(y)
        y = self.act2(y)
        before_pool = y
        y = self.pool(y)
        return y, before_pool


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
