import torch
import torch.nn as nn


class ConcatSquashConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0,
        dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatSquashConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(dim_in, dim_out, kernel_size=ksize, stride=
            stride, padding=padding, dilation=dilation, groups=groups, bias
            =bias)
        self._hyper_gate = nn.Linear(1, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(1, 1))
            ).view(1, -1, 1, 1) + self._hyper_bias(t.view(1, 1)).view(1, -1,
            1, 1)


def get_inputs():
    return [torch.rand([1, 1]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_in': 4, 'dim_out': 4}]
