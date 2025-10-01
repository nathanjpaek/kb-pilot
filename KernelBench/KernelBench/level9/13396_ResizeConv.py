import torch
from torch import nn
import torch.utils.data


def get_conv(dim=3):
    """Chooses an implementation for a convolution layer."""
    if dim == 3:
        return nn.Conv3d
    elif dim == 2:
        return nn.Conv2d
    else:
        raise ValueError('dim has to be 2 or 3')


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


def conv1(in_channels, out_channels, dim=3):
    """Returns a 1x1 or 1x1x1 convolution, depending on dim"""
    return get_conv(dim)(in_channels, out_channels, kernel_size=1)


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


class ResizeConv(nn.Module):
    """Upsamples by 2x and applies a convolution.

    This is meant as a replacement for transposed convolution to avoid
    checkerboard artifacts. See

    - https://distill.pub/2016/deconv-checkerboard/
    - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, planar=
        False, dim=3, upsampling_mode='nearest'):
        super().__init__()
        self.upsampling_mode = upsampling_mode
        self.scale_factor = 2
        if dim == 3 and planar:
            self.scale_factor = planar_kernel(self.scale_factor)
        self.dim = dim
        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode=
            self.upsampling_mode)
        if kernel_size == 3:
            self.conv = conv3(in_channels, out_channels, padding=1, planar=
                planar, dim=dim)
        elif kernel_size == 1:
            self.conv = conv1(in_channels, out_channels, dim=dim)
        else:
            raise ValueError(
                f'kernel_size={kernel_size} is not supported. Choose 1 or 3.')

    def forward(self, x):
        return self.conv(self.upsample(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
