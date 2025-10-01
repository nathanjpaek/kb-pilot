import torch
import torch.nn as nn
import torch.utils


class Conv2d(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, padding):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=
            1, padding=padding)

    def forward(self, x):
        return self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'padding': 4}]
