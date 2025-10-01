import torch
import torch.nn as nn


class MaxPool(nn.Module):

    def __init__(self, kernel_size, stride=1, padding=1, zero_pad=False):
        super(MaxPool, self).__init__()
        self.is_zero_padded = zero_pad
        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        if self.is_zero_padded:
            x = self.zero_pad(x)
        x = self.pool(x)
        if self.is_zero_padded:
            x = x[:, :, 1:, 1:]
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4}]
