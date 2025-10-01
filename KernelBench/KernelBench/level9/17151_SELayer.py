import math
import torch
from torch import nn
from torch.nn import functional as F
import torch.onnx
from torch.optim.lr_scheduler import *


def composite_swish(inputs_1, inputs_2):
    return inputs_1 * torch.sigmoid(inputs_2)


def swish(x):
    return torch.sigmoid(x) * x


class _Conv2dSamePadding(nn.Conv2d):
    """ Class implementing 2d adaptively padded convolutions """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0,
            dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]
            ] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] +
            1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] +
            1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h -
                pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.
            padding, self.dilation, self.groups)


class SELayer(nn.Module):
    """
    Class impements the squeeze and excitation layer
    """

    def __init__(self, in_channels, se_ratio):
        super(SELayer, self).__init__()
        assert se_ratio <= 1.0 and se_ratio >= 0.0, 'se_ratio should be in [0,1]'
        num_squeezed_channels = max(1, int(in_channels * se_ratio))
        self._se_reduce = _Conv2dSamePadding(in_channels,
            num_squeezed_channels, 1)
        self._se_expand = _Conv2dSamePadding(num_squeezed_channels,
            in_channels, 1)

    def forward(self, inputs):
        inputs_squeezed = F.adaptive_avg_pool2d(inputs, 1)
        inputs_squeezed = self._se_expand(swish(self._se_reduce(
            inputs_squeezed)))
        return composite_swish(inputs, inputs_squeezed)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'se_ratio': 0}]
