import math
import torch
from torch import nn
from typing import List
from typing import Union
import torch.nn.functional as F
from typing import Optional
from typing import Tuple
from torch.nn.common_types import _size_2_t


def get_same_padding(x: 'int', k: 'int', s: 'int', d: 'int') ->int:
    """
    Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
    Args:
        x(`Int`):
            Input tensor shape.
        k(`Int`):
            Convolution kernel size.
        s(`Int`):
            Convolution stride parameter.
        d(`Int`):
            Convolution dilation parameter.
    Returns:
        (`Int`):
            Padding value for 'SAME' padding.
    """
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


def pad_same(x: 'torch.Tensor', k: 'List[int]', s: 'List[int]', d:
    'List[int]'=(1, 1), value: 'float'=0) ->torch.Tensor:
    """
    Dynamically pad input x with 'SAME' padding for conv with specified args
    Args:
        x(`torch.Tensor`):
            Input tensor.
        k(`List[Int]`):
            Convolution kernel sizes.
        s(`List[Int]`):
            Convolution stride parameters.
        d(`List[Int]`):
            Convolution dilation parameter.
        value(`Float`):
            Value for padding.
    Returns:
        (`torch.Tensor`):
            Output Tensor for conv with 'SAME' padding.
    """
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw,
        k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - 
            pad_h // 2], value=value)
    return x


def conv2d_same(x: 'torch.Tensor', weight: 'torch.Tensor', bias:
    'Optional[torch.Tensor]'=None, stride: 'Tuple[int, int]'=(1, 1),
    padding: 'Tuple[int, int]'=(0, 0), dilation: 'Tuple[int, int]'=(1, 1),
    groups: 'int'=1):
    """
    Tensorflow like 'SAME' convolution function for 2D convolutions.
    """
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    _ = padding
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    """
    Tensorflow like 'SAME' convolution wrapper for 2D convolutions.

    Args:
        in_channels (`int`):
            Number of channels in the input image.
        out_channels (`int`):
            Number of channels produced by the convolution.
        kernel_size (`Union[int, Tuple]`):
            Size of the convolving kernel.
        stride (`Union[int, Tuple]`):
            Stride of the convolution.
        padding (`Union[int, Tuple, str]`):
            Padding added to all four sides of the input.
        dilation (`int`):
            Spacing between kernel elements.
        groups (`int`):
            Number of blocked connections from input channels to output channels.
        bias (`bool`):
            If True, adds a learnable bias to the output.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size:
        '_size_2_t', stride: '_size_2_t'=1, padding:
        'Union[str, _size_2_t]'=0, dilation: '_size_2_t'=1, groups: 'int'=1,
        bias: 'bool'=True) ->None:
        super().__init__(in_channels, out_channels, kernel_size, stride, 0,
            dilation, groups, bias)
        _ = padding

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return conv2d_same(x, self.weight, self.bias, self.stride, self.
            padding, self.dilation, self.groups)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
