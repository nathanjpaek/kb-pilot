import torch
from torch.nn import functional as F
import torch.nn as nn


def tf_2xupsample_bilinear(x):
    b, c, h, w = x.shape
    out = torch.zeros(b, c, h * 2, w * 2)
    out[:, :, ::2, ::2] = x
    padded = F.pad(x, (0, 1, 0, 1), mode='replicate')
    out[:, :, 1::2, ::2] = (padded[:, :, :-1, :-1] + padded[:, :, 1:, :-1]) / 2
    out[:, :, ::2, 1::2] = (padded[:, :, :-1, :-1] + padded[:, :, :-1, 1:]) / 2
    out[:, :, 1::2, 1::2] = (padded[:, :, :-1, :-1] + padded[:, :, 1:, 1:]) / 2
    return out


def tf_same_padding(x, k_size=3):
    j = k_size // 2
    return F.pad(x, (j - 1, j, j - 1, j))


class Upsample(nn.Module):
    """Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.

    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.

    Args:
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
            output spatial sizes
        scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):
            multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        align_corners (bool, optional): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is
            ``'linear'``, ``'bilinear'``, or ``'trilinear'``. Default: ``False``
    """

    def __init__(self, size=None, scale_factor=None, mode='nearest',
        align_corners=None):
        super(Upsample, self).__init__()
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.size = size
        self.align_corners = align_corners

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=
            self.scale_factor, mode=self.mode, align_corners=self.align_corners
            )

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


class ResBlock(nn.Module):

    def __init__(self, in_nf, out_nf=32, slope=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_nf, out_nf, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_nf, out_nf, 3, 1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, inputs):
        x = self.conv2(self.leaky_relu(self.conv1(inputs)))
        return x + inputs


class Upsample_2xBil_TF(nn.Module):

    def __init__(self):
        super(Upsample_2xBil_TF, self).__init__()

    def forward(self, x):
        return tf_2xupsample_bilinear(x)


class UnetGeneratorWBC(nn.Module):
    """ UNet Generator as used in Learning to Cartoonize Using White-box
    Cartoon Representations for image to image translation
    https://systemerrorwang.github.io/White-box-Cartoonization/paper/06791.pdf
    https://systemerrorwang.github.io/White-box-Cartoonization/paper/06791-supp.pdf
    """

    def __init__(self, nf=32, mode='pt', slope=0.2):
        super(UnetGeneratorWBC, self).__init__()
        self.mode = mode
        self.conv = nn.Conv2d(3, nf, 7, 1, padding=3)
        if mode == 'tf':
            self.conv_1 = nn.Conv2d(nf, nf, 3, stride=2, padding=0)
        else:
            self.conv_1 = nn.Conv2d(nf, nf, 3, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(nf, nf * 2, 3, 1, padding=1)
        if mode == 'tf':
            self.conv_3 = nn.Conv2d(nf * 2, nf * 2, 3, stride=2, padding=0)
        else:
            self.conv_3 = nn.Conv2d(nf * 2, nf * 2, 3, stride=2, padding=1)
        self.conv_4 = nn.Conv2d(nf * 2, nf * 4, 3, 1, padding=1)
        self.block_0 = ResBlock(nf * 4, nf * 4, slope=slope)
        self.block_1 = ResBlock(nf * 4, nf * 4, slope=slope)
        self.block_2 = ResBlock(nf * 4, nf * 4, slope=slope)
        self.block_3 = ResBlock(nf * 4, nf * 4, slope=slope)
        self.conv_5 = nn.Conv2d(nf * 4, nf * 2, 3, 1, padding=1)
        self.conv_6 = nn.Conv2d(nf * 2, nf * 2, 3, 1, padding=1)
        self.conv_7 = nn.Conv2d(nf * 2, nf, 3, 1, padding=1)
        self.conv_8 = nn.Conv2d(nf, nf, 3, 1, padding=1)
        self.conv_9 = nn.Conv2d(nf, 3, 7, 1, padding=3)
        self.leaky_relu = nn.LeakyReLU(negative_slope=slope, inplace=False)
        if mode == 'tf':
            self.upsample = Upsample_2xBil_TF()
        else:
            self.upsample = Upsample(scale_factor=2, mode='bilinear',
                align_corners=False)

    def forward(self, x):
        x0 = self.conv(x)
        x0 = self.leaky_relu(x0)
        if self.mode == 'tf':
            x1 = self.conv_1(tf_same_padding(x0))
        else:
            x1 = self.conv_1(x0)
        x1 = self.leaky_relu(x1)
        x1 = self.conv_2(x1)
        x1 = self.leaky_relu(x1)
        if self.mode == 'tf':
            x2 = self.conv_3(tf_same_padding(x1))
        else:
            x2 = self.conv_3(x1)
        x2 = self.leaky_relu(x2)
        x2 = self.conv_4(x2)
        x2 = self.leaky_relu(x2)
        x2 = self.block_3(self.block_2(self.block_1(self.block_0(x2))))
        x2 = self.conv_5(x2)
        x2 = self.leaky_relu(x2)
        x3 = self.upsample(x2)
        x3 = self.conv_6(x3 + x1)
        x3 = self.leaky_relu(x3)
        x3 = self.conv_7(x3)
        x3 = self.leaky_relu(x3)
        x4 = self.upsample(x3)
        x4 = self.conv_8(x4 + x0)
        x4 = self.leaky_relu(x4)
        x4 = self.conv_9(x4)
        return x4


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
