import torch
import torch.nn.functional as F
from functools import partial
import torch.nn as nn


def dispatcher(dispatch_fn):

    def decorated(key, *args):
        if callable(key):
            return key
        if key is None:
            key = 'none'
        return dispatch_fn(key, *args)
    return decorated


@dispatcher
def activ_dispatch(activ):
    return {'none': nn.Identity, 'relu': nn.ReLU, 'lrelu': partial(nn.
        LeakyReLU, negative_slope=0.2)}[activ.lower()]


@dispatcher
def norm_dispatch(norm):
    return {'none': nn.Identity, 'in': partial(nn.InstanceNorm2d, affine=
        False), 'bn': nn.BatchNorm2d}[norm.lower()]


@dispatcher
def pad_dispatch(pad_type):
    return {'zero': nn.ZeroPad2d, 'replicate': nn.ReplicationPad2d,
        'reflect': nn.ReflectionPad2d}[pad_type.lower()]


def spectral_norm(module):
    """ init & apply spectral norm """
    nn.init.xavier_uniform_(module.weight, 2 ** 0.5)
    if hasattr(module, 'bias') and module.bias is not None:
        module.bias.data.zero_()
    return nn.utils.spectral_norm(module)


@dispatcher
def w_norm_dispatch(w_norm):
    return {'spectral': spectral_norm, 'none': lambda x: x}[w_norm.lower()]


class ConvBlock(nn.Module):
    """Pre-activate conv block"""

    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=1,
        norm='none', activ='relu', bias=True, upsample=False, downsample=
        False, w_norm='none', pad_type='zero', dropout=0.0):
        if kernel_size == 1:
            assert padding == 0
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        activ = activ_dispatch(activ)
        norm = norm_dispatch(norm)
        w_norm = w_norm_dispatch(w_norm)
        pad = pad_dispatch(pad_type)
        self.upsample = upsample
        self.downsample = downsample
        self.norm = norm(C_in)
        self.activ = activ()
        if dropout > 0.0:
            self.dropout = nn.Dropout2d(p=dropout)
        self.pad = pad(padding)
        self.conv = w_norm(nn.Conv2d(C_in, C_out, kernel_size, stride, bias
            =bias))

    def forward(self, x):
        x = self.norm(x)
        x = self.activ(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.conv(self.pad(x))
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x


class ResBlock(nn.Module):
    """Pre-activate residual block"""

    def __init__(self, C_in, C_out, kernel_size=3, padding=1, upsample=
        False, downsample=False, norm='none', w_norm='none', activ='relu',
        pad_type='zero', dropout=0.0):
        assert not (upsample and downsample)
        super().__init__()
        w_norm = w_norm_dispatch(w_norm)
        self.C_in = C_in
        self.C_out = C_out
        self.upsample = upsample
        self.downsample = downsample
        self.conv1 = ConvBlock(C_in, C_out, kernel_size, 1, padding, norm,
            activ, upsample=upsample, w_norm=w_norm, pad_type=pad_type,
            dropout=dropout)
        self.conv2 = ConvBlock(C_out, C_out, kernel_size, 1, padding, norm,
            activ, w_norm=w_norm, pad_type=pad_type, dropout=dropout)
        if C_in != C_out or upsample or downsample:
            self.skip = w_norm(nn.Conv2d(C_in, C_out, 1))

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.conv2(out)
        if self.downsample:
            out = F.avg_pool2d(out, 2)
        if hasattr(self, 'skip'):
            if self.upsample:
                x = F.interpolate(x, scale_factor=2)
            x = self.skip(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)
        out = out + x
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'C_in': 4, 'C_out': 4}]
