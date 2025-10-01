from torch.nn import Module
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module


class my_AvgPool2d(Module):

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
        count_include_pad=True):
        super(my_AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input):
        input = input.transpose(3, 1)
        input = F.avg_pool2d(input, self.kernel_size, self.stride, self.
            padding, self.ceil_mode, self.count_include_pad)
        input = input.transpose(3, 1).contiguous()
        return input

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'kernel_size=' + str(self.
            kernel_size) + ', stride=' + str(self.stride) + ', padding=' + str(
            self.padding) + ', ceil_mode=' + str(self.ceil_mode
            ) + ', count_include_pad=' + str(self.count_include_pad) + ')'


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4}]
