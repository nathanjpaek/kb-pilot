from torch.nn import Module
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair


class my_MaxPool2d(Module):

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
        return_indices=False, ceil_mode=False):
        super(my_MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        input = input.transpose(3, 1)
        input = F.max_pool2d(input, self.kernel_size, self.stride, self.
            padding, self.dilation, self.ceil_mode, self.return_indices)
        input = input.transpose(3, 1).contiguous()
        return input

    def __repr__(self):
        kh, kw = _pair(self.kernel_size)
        dh, dw = _pair(self.stride)
        padh, padw = _pair(self.padding)
        dilh, dilw = _pair(self.dilation)
        padding_str = ', padding=(' + str(padh) + ', ' + str(padw
            ) + ')' if padh != 0 or padw != 0 else ''
        dilation_str = ', dilation=(' + str(dilh) + ', ' + str(dilw
            ) + ')' if dilh != 0 and dilw != 0 else ''
        ceil_str = ', ceil_mode=' + str(self.ceil_mode)
        return self.__class__.__name__ + '(' + 'kernel_size=(' + str(kh
            ) + ', ' + str(kw) + ')' + ', stride=(' + str(dh) + ', ' + str(dw
            ) + ')' + padding_str + dilation_str + ceil_str + ')'


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4}]
