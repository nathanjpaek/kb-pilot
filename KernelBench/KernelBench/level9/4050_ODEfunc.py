import torch
import torch.nn as nn


def norm(dim):
    """
    Group normalization to improve model accuracy and training speed.
    """
    return nn.GroupNorm(min(1, dim), dim)


class ConcatConv1d(nn.Module):
    """
    1d convolution concatenated with time for usage in ODENet.
    """

    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=0,
        bias=True, transpose=False):
        super(ConcatConv1d, self).__init__()
        module = nn.ConvTranspose1d if transpose else nn.Conv1d
        self._layer = module(dim_in + 1, dim_out, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=bias)

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):
    """
    Network architecture for ODENet.
    """

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv1d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv1d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


def get_inputs():
    return [torch.rand([4, 1, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
