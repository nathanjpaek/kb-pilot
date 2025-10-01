import torch
import torch.nn as nn


def set_activate_layer(types):
    if types == 'relu':
        activation = nn.ReLU()
    elif types == 'lrelu':
        activation = nn.LeakyReLU(0.2)
    elif types == 'tanh':
        activation = nn.Tanh()
    elif types == 'sig':
        activation = nn.Sigmoid()
    elif types == 'none':
        activation = None
    else:
        assert 0, f'Unsupported activation: {types}'
    return activation


def set_norm_layer(norm_type, norm_dim):
    if norm_type == 'bn':
        norm = nn.BatchNorm2d(norm_dim)
    elif norm_type == 'in':
        norm = nn.InstanceNorm2d(norm_dim)
    elif norm_type == 'none':
        norm = None
    else:
        assert 0, 'Unsupported normalization: {}'.format(norm)
    return norm


class Interpolate(nn.Module):

    def __init__(self, scale_factor, mode='bilinear'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode,
            align_corners=False)
        return x


class ResBlock(nn.Module):

    def __init__(self, in_c, out_c, scale_factor=1, norm='in', activation=
        'lrelu'):
        super(ResBlock, self).__init__()
        self.norm1 = set_norm_layer(norm, out_c)
        self.norm2 = set_norm_layer(norm, out_c)
        self.activ = set_activate_layer(activation)
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_c, out_channels=out_c,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1x1 = nn.Conv2d(in_channels=in_c, out_channels=out_c,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.resize = Interpolate(scale_factor=scale_factor)

    def forward(self, feat):
        feat1 = self.norm1(feat)
        feat1 = self.activ(feat1)
        feat1 = self.conv1(feat1)
        feat1 = self.resize(feat1)
        feat1 = self.norm2(feat1)
        feat1 = self.activ(feat1)
        feat1 = self.conv2(feat1)
        feat2 = self.conv1x1(feat)
        feat2 = self.resize(feat2)
        return feat1 + feat2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_c': 4, 'out_c': 4}]
