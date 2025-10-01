import math
import torch
import torch.nn.functional as F


class PadSameConv2d(torch.nn.Module):

    def __init__(self, kernel_size, stride=1):
        """
        Imitates padding_mode="same" from tensorflow.
        :param kernel_size: Kernelsize of the convolution, int or tuple/list
        :param stride: Stride of the convolution, int or tuple/list
        """
        super().__init__()
        if isinstance(kernel_size, (tuple, list)):
            self.kernel_size_y = kernel_size[0]
            self.kernel_size_x = kernel_size[1]
        else:
            self.kernel_size_y = kernel_size
            self.kernel_size_x = kernel_size
        if isinstance(stride, (tuple, list)):
            self.stride_y = stride[0]
            self.stride_x = stride[1]
        else:
            self.stride_y = stride
            self.stride_x = stride

    def forward(self, x: 'torch.Tensor'):
        _, _, height, width = x.shape
        padding_y = (self.stride_y * (math.ceil(height / self.stride_y) - 1
            ) + self.kernel_size_y - height) / 2
        padding_x = (self.stride_x * (math.ceil(width / self.stride_x) - 1) +
            self.kernel_size_x - width) / 2
        padding = [math.floor(padding_x), math.ceil(padding_x), math.floor(
            padding_y), math.ceil(padding_y)]
        return F.pad(input=x, pad=padding)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4}]
