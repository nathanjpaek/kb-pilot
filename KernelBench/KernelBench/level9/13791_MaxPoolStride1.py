import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import torch._utils


class MaxPoolStride1(nn.Module):

    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x):
        padding = int(self.pad / 2)
        padded_x = F.pad(x, (padding, padding, padding, padding), mode=
            'constant', value=0)
        pooled_x = nn.MaxPool2d(self.kernel_size, 1)(padded_x)
        return pooled_x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4}]
