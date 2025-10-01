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


def get_padding(conv_mode, kernel_size):
    if conv_mode == 'valid' or kernel_size == 1:
        return 0
    elif conv_mode == 'same' and kernel_size == 3:
        return 1
    else:
        raise NotImplementedError(
            f'conv_mode {conv_mode} with kernel_size {kernel_size} unsupported.'
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


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, planar=
        False, activation='relu', normalization=None, dim=3, conv_mode=
        'same', residual=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.conv_mode = conv_mode
        self.activation = activation
        self.residual = residual
        self.dim = dim
        padding = get_padding(conv_mode, kernel_size)
        if planar:
            padding = planar_pad(padding)
            kernel_size = planar_kernel(kernel_size)
        conv_class = get_conv(dim)
        self.conv1 = conv_class(in_channels, out_channels, kernel_size=
            kernel_size, padding=padding)
        self.norm1 = get_normalization(normalization, self.out_channels,
            dim=dim)
        self.act1 = get_activation(activation)
        self.conv2 = conv_class(out_channels, out_channels, kernel_size=
            kernel_size, padding=padding)
        self.norm2 = get_normalization(normalization, self.out_channels,
            dim=dim)
        self.act2 = get_activation(activation)
        if self.residual and self.in_channels != self.out_channels:
            self.proj = conv_class(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

    def forward(self, inp):
        y = self.conv1(inp)
        y = self.norm1(y)
        y = self.act1(y)
        y = self.conv2(y)
        if self.residual:
            y += self.proj(inp)
        y = self.norm2(y)
        y = self.act2(y)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
