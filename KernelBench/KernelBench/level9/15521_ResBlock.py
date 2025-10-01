from _paritybench_helpers import _mock_config
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional


def Activation_layer(activation_cfg, inplace=True):
    out = None
    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    else:
        out = nn.LeakyReLU(negative_slope=0.01, inplace=inplace)
    return out


def Norm_layer(norm_cfg, inplanes):
    out = None
    if norm_cfg == 'BN':
        out = nn.BatchNorm3d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    else:
        out = nn.InstanceNorm3d(inplanes, affine=True)
    return out


def conv3x3x3(in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding
    =(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=False, weight_std=False):
    """3x3x3 convolution with padding"""
    if weight_std:
        return Conv3d_wd(in_planes, out_planes, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=
            groups, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=
            groups, bias=bias)


class Conv3d_wd(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1,
        1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=False):
        super(Conv3d_wd, self).__init__(in_channels, out_channels,
            kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True
            ).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) +
            1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups)


class Conv3dBlock(nn.Module):

    def __init__(self, in_channels, out_channels, norm_cfg, activation_cfg,
        kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1
        ), bias=False, weight_std=False):
        super(Conv3dBlock, self).__init__()
        self.conv = conv3x3x3(in_channels, out_channels, kernel_size=
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            bias=bias, weight_std=weight_std)
        self.norm = Norm_layer(norm_cfg, out_channels)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlin(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, inplanes, planes, norm_cfg, activation_cfg,
        weight_std=False):
        super(ResBlock, self).__init__()
        self.resconv1 = Conv3dBlock(inplanes, planes, norm_cfg,
            activation_cfg, kernel_size=3, stride=1, padding=1, bias=False,
            weight_std=weight_std)
        self.resconv2 = Conv3dBlock(planes, planes, norm_cfg,
            activation_cfg, kernel_size=3, stride=1, padding=1, bias=False,
            weight_std=weight_std)

    def forward(self, x):
        residual = x
        out = self.resconv1(x)
        out = self.resconv2(out)
        out = out + residual
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'planes': 4, 'norm_cfg': _mock_config(),
        'activation_cfg': _mock_config()}]
