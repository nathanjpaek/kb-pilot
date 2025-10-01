import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair


def same_padding_length(input_length, filter_size, stride, dilation=1):
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    output_length = (input_length + stride - 1) // stride
    pad_length = max(0, (output_length - 1) * stride + dilated_filter_size -
        input_length)
    return pad_length


def compute_same_padding2d(input_shape, kernel_size, strides, dilation):
    space = input_shape[2:]
    assert len(space) == 2, '{}'.format(space)
    new_space = []
    new_input = []
    for i in range(len(space)):
        pad_length = same_padding_length(space[i], kernel_size[i], stride=
            strides[i], dilation=dilation[i])
        new_space.append(pad_length)
        new_input.append(pad_length % 2)
    return tuple(new_space), tuple(new_input)


class _Conv2d_dilated(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)
        super(_Conv2d_dilated, self).__init__(in_channels, out_channels,
            kernel_size, stride, _pair(0), dilation, False, _pair(0),
            groups, bias, padding_mode='zeros')

    def forward(self, input, dilation=None):
        input_shape = list(input.size())
        dilation_rate = self.dilation if dilation is None else _pair(dilation)
        padding, pad_input = compute_same_padding2d(input_shape,
            kernel_size=self.kernel_size, strides=self.stride, dilation=
            dilation_rate)
        if pad_input[0] == 1 or pad_input[1] == 1:
            input = F.pad(input, [0, int(pad_input[0]), 0, int(pad_input[1])])
        return F.conv2d(input, self.weight, self.bias, self.stride, (
            padding[0] // 2, padding[1] // 2), dilation_rate, self.groups)


class Conv2d_dilated(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL
        ='relu', same_padding=False, dilation=1, bn=False, bias=True, groups=1
        ):
        super(Conv2d_dilated, self).__init__()
        self.conv = _Conv2d_dilated(in_channels, out_channels, kernel_size,
            stride, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0,
            affine=True) if bn else None
        if NL == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        elif NL == 'tanh':
            self.relu = nn.Tanh()
        elif NL == 'lrelu':
            self.relu = nn.LeakyReLU(inplace=True)
        elif NL == 'sigmoid':
            self.relu = nn.Sigmoid()
        else:
            self.relu = None

    def forward(self, x, dilation=None):
        x = self.conv(x, dilation)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
