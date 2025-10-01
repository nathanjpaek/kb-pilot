import torch
from torch import nn
import torch.nn.functional as F


def conv2d_same_padding(input, weight, bias=None, stride=1, dilation=1,
    groups=1):
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
        effective_filter_size_rows - input_rows)
    rows_odd = padding_rows % 2 != 0
    input_cols = input.size(3)
    filter_cols = weight.size(3)
    effective_filter_size_cols = (filter_cols - 1) * dilation[1] + 1
    out_cols = (input_cols + stride[1] - 1) // stride[1]
    padding_cols = max(0, (out_cols - 1) * stride[1] +
        effective_filter_size_cols - input_cols)
    cols_odd = padding_cols % 2 != 0
    if rows_odd or cols_odd:
        input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])
    return F.conv2d(input, weight, bias, stride, padding=(padding_rows // 2,
        padding_cols // 2), dilation=dilation, groups=groups)


class Conv2dSamePadding(nn.Conv2d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867

    This solution is mostly copied from
    https://github.com/pytorch/pytorch/issues/3867#issuecomment-349279036

    Note that the padding argument in the initializer doesn't do anything now
    """

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.
            stride, self.dilation, self.groups)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
