import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_same_padding(in_size, kernel_size, stride):
    """'Same 'same' operation with tensorflow
    notice：padding=(0, 1, 0, 1) and padding=(1, 1, 1, 1) are different

    padding=(1, 1, 1, 1):
        out(H, W) = (in + [2 * padding] − kernel_size) // stride + 1

    'same' padding=(0, 1, 0, 1):
        out(H, W) = (in + [2 * padding] − kernel_size) / stride + 1

    :param in_size: Union[int, tuple(in_h, in_w)]
    :param kernel_size: Union[int, tuple(kernel_h, kernel_w)]
    :param stride: Union[int, tuple(stride_h, stride_w)]
    :return: padding: tuple(left, right, top, bottom)
    """
    in_h, in_w = (in_size, in_size) if isinstance(in_size, int) else in_size
    kernel_h, kernel_w = (kernel_size, kernel_size) if isinstance(kernel_size,
        int) else kernel_size
    stride_h, stride_w = (stride, stride) if isinstance(stride, int
        ) else stride
    out_h, out_w = math.ceil(in_h / stride_h), math.ceil(in_w / stride_w)
    pad_h = max((out_h - 1) * stride_h + kernel_h - in_h, 0)
    pad_w = max((out_w - 1) * stride_w + kernel_w - in_w, 0)
    return pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2


class MaxPool2dSamePadding(nn.MaxPool2d):
    """MaxPool2dDynamicSamePadding

    由于输入大小都是128的倍数，所以动态池化和静态池化的结果是一致的。此处用动态池化代替静态池化，因为实现方便。

    Since the input size is a multiple of 128,
    the results of dynamic maxpool and static maxpool are consistent.
    Here, dynamic maxpool is used instead of static maxpool,
    because it is convenient to implement"""

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        super(MaxPool2dSamePadding, self).__init__(kernel_size, stride)

    def forward(self, x):
        padding = get_same_padding(x.shape[-2:], self.kernel_size, self.stride)
        x = F.pad(x, padding)
        x = super().forward(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4, 'stride': 1}]
