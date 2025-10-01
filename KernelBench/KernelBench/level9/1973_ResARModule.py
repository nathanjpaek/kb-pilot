import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm


def build_norm_layer(norm_type, param=None, num_feats=None):
    if norm_type == 'bnorm':
        return nn.BatchNorm1d(num_feats)
    elif norm_type == 'snorm':
        spectral_norm(param)
        return None
    elif norm_type is None:
        return None
    else:
        raise TypeError('Unrecognized norm type: ', norm_type)


class ResARModule(nn.Module):

    def __init__(self, ninp, fmaps, res_fmaps, kwidth, dilation, bias=True,
        norm_type=None, act=None):
        super().__init__()
        self.dil_conv = nn.Conv1d(ninp, fmaps, kwidth, dilation=dilation,
            bias=bias)
        if act is not None:
            self.act = getattr(nn, act)()
        else:
            self.act = nn.PReLU(fmaps, init=0)
        self.dil_norm = build_norm_layer(norm_type, self.dil_conv, fmaps)
        self.kwidth = kwidth
        self.dilation = dilation
        self.conv_1x1_skip = nn.Conv1d(fmaps, ninp, 1, bias=bias)
        self.conv_1x1_skip_norm = build_norm_layer(norm_type, self.
            conv_1x1_skip, ninp)
        self.conv_1x1_res = nn.Conv1d(fmaps, res_fmaps, 1, bias=bias)
        self.conv_1x1_res_norm = build_norm_layer(norm_type, self.
            conv_1x1_res, res_fmaps)

    def forward_norm(self, x, norm_layer):
        if norm_layer is not None:
            return norm_layer(x)
        else:
            return x

    def forward(self, x):
        kw__1 = self.kwidth - 1
        P = kw__1 + kw__1 * (self.dilation - 1)
        x_p = F.pad(x, (P, 0))
        h = self.dil_conv(x_p)
        h = self.forward_norm(h, self.dil_norm)
        h = self.act(h)
        a = h
        h = self.conv_1x1_skip(h)
        h = self.forward_norm(h, self.conv_1x1_skip_norm)
        y = x + h
        sh = self.conv_1x1_res(a)
        sh = self.forward_norm(sh, self.conv_1x1_res_norm)
        return y, sh


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'ninp': 4, 'fmaps': 4, 'res_fmaps': 4, 'kwidth': 4,
        'dilation': 1}]
