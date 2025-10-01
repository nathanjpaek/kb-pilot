import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def pixel_norm(x):
    sigma = x.norm(dim=1, keepdim=True)
    out = x / (sigma + 1e-05)
    return out


class EqualizedLR(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        self._make_params()

    def _make_params(self):
        weight = self.module.weight
        height = weight.data.shape[0]
        width = weight.view(height, -1).data.shape[1]
        del self.module._parameters['weight']
        self.module.weight = None
        self.weight = nn.Parameter(weight.data)
        self.factor = np.sqrt(2 / width)
        nn.init.normal_(self.weight)
        self.bias = self.module.bias
        self.module.bias = None
        if self.bias is not None:
            del self.module._parameters['bias']
            nn.init.zeros_(self.bias)

    def forward(self, *args, **kwargs):
        self.module.weight = self.factor * self.weight
        if self.bias is not None:
            self.module.bias = 1.0 * self.bias
        out = self.module.forward(*args, **kwargs)
        self.module.weight = None
        self.module.bias = None
        return out


class ResnetBlockConv2d(nn.Module):

    def __init__(self, f_in, f_out=None, f_hidden=None, is_bias=True, actvn
        =F.relu, factor=1.0, eq_lr=False, pixel_norm=False):
        super().__init__()
        if f_out is None:
            f_out = f_in
        if f_hidden is None:
            f_hidden = min(f_in, f_out)
        self.f_in = f_in
        self.f_hidden = f_hidden
        self.f_out = f_out
        self.factor = factor
        self.eq_lr = eq_lr
        self.use_pixel_norm = pixel_norm
        self.actvn = actvn
        self.conv_0 = nn.Conv2d(self.f_in, self.f_hidden, 3, stride=1,
            padding=1)
        self.conv_1 = nn.Conv2d(self.f_hidden, self.f_out, 3, stride=1,
            padding=1, bias=is_bias)
        if self.eq_lr:
            self.conv_0 = EqualizedLR(self.conv_0)
            self.conv_1 = EqualizedLR(self.conv_1)
        if f_in == f_out:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Conv2d(f_in, f_out, 1, bias=False)
            if self.eq_lr:
                self.shortcut = EqualizedLR(self.shortcut)
        nn.init.zeros_(self.conv_1.weight)

    def forward(self, x):
        x_s = self.shortcut(x)
        if self.use_pixel_norm:
            x = pixel_norm(x)
        dx = self.conv_0(self.actvn(x))
        if self.use_pixel_norm:
            dx = pixel_norm(dx)
        dx = self.conv_1(self.actvn(dx))
        out = x_s + self.factor * dx
        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'f_in': 4}]
