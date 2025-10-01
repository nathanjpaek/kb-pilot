import torch
import torch.nn as nn


class BlendConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0,
        dilation=1, groups=1, bias=True, transpose=False, **unused_kwargs):
        super(BlendConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer0 = module(dim_in, dim_out, kernel_size=ksize, stride=
            stride, padding=padding, dilation=dilation, groups=groups, bias
            =bias)
        self._layer1 = module(dim_in, dim_out, kernel_size=ksize, stride=
            stride, padding=padding, dilation=dilation, groups=groups, bias
            =bias)

    def forward(self, t, x):
        y0 = self._layer0(x)
        y1 = self._layer1(x)
        return y0 + (y1 - y0) * t


def get_inputs():
    return [torch.rand([4, 4, 2, 2]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_in': 4, 'dim_out': 4}]
