import torch
import torch.utils.data
import torch.utils.data.distributed
from torch import nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim


def _calc_same_pad(input_: 'int', kernel: 'int', stride: 'int', dilation: 'int'
    ):
    """calculate same padding"""
    return max((-(input_ // -stride) - 1) * stride + (kernel - 1) *
        dilation + 1 - input_, 0)


def _same_pad_arg(input_size, kernel_size, stride, dilation):
    input_height, input_width = input_size
    kernel_height, kernel_width = kernel_size
    pad_h = _calc_same_pad(input_height, kernel_height, stride[0], dilation[0])
    pad_w = _calc_same_pad(input_width, kernel_width, stride[1], dilation[1])
    return [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]


class Conv2dSameExport(nn.Conv2d):
    """ ONNX export friendly Tensorflow like 'SAME' convolution wrapper for 2D convolutions

    NOTE: This does not currently work with torch.jit.script
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0,
            dilation, groups, bias)
        self.pad = None
        self.pad_input_size = 0, 0

    def forward(self, input_):
        input_size = input_.size()[-2:]
        if self.pad is None:
            pad_arg = _same_pad_arg(input_size, self.weight.size()[-2:],
                self.stride, self.dilation)
            self.pad = nn.ZeroPad2d(pad_arg)
            self.pad_input_size = input_size
        if self.pad is not None:
            input_ = self.pad(input_)
        return F.conv2d(input_, self.weight, self.bias, self.stride, self.
            padding, self.dilation, self.groups)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
