import torch
import torch.cuda
from torch.nn import functional as F
from torch import nn
import torch.distributed
from torch.cuda.amp import autocast as autocast
import torch.utils.data
import torch.optim


class ConvNorm(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super().__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            bias=bias)
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.
            calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class BottleneckLayerLayer(nn.Module):

    def __init__(self, in_dim, reduction_factor, norm='weightnorm',
        non_linearity='relu', use_pconv=False):
        super(BottleneckLayerLayer, self).__init__()
        self.reduction_factor = reduction_factor
        reduced_dim = int(in_dim / reduction_factor)
        self.out_dim = reduced_dim
        if self.reduction_factor > 1:
            fn = ConvNorm(in_dim, reduced_dim, kernel_size=3)
            if norm == 'weightnorm':
                fn = torch.nn.utils.weight_norm(fn.conv, name='weight')
            elif norm == 'instancenorm':
                fn = nn.Sequential(fn, nn.InstanceNorm1d(reduced_dim,
                    affine=True))
            self.projection_fn = fn
            self.non_linearity = non_linearity

    def forward(self, x):
        if self.reduction_factor > 1:
            x = self.projection_fn(x)
            if self.non_linearity == 'relu':
                x = F.relu(x)
            elif self.non_linearity == 'leakyrelu':
                x = F.leaky_relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'reduction_factor': 4}]
