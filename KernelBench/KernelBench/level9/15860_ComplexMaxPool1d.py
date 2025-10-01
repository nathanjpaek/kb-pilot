import torch
from torch import nn
import torch.utils


class ComplexMaxPool1d(nn.Module):

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
        return_indices=False, ceil_mode=False):
        super(ComplexMaxPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices
        self.maxpool_r = nn.MaxPool1d(kernel_size=self.kernel_size, stride=
            self.stride, padding=self.padding, dilation=self.dilation,
            ceil_mode=self.ceil_mode, return_indices=self.return_indices)
        self.maxpool_i = nn.MaxPool1d(kernel_size=self.kernel_size, stride=
            self.stride, padding=self.padding, dilation=self.dilation,
            ceil_mode=self.ceil_mode, return_indices=self.return_indices)

    def forward(self, input_r, input_i):
        return self.maxpool_r(input_r), self.maxpool_i(input_i)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4}]
