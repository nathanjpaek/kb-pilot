import torch
import torch.utils.data
import torch.nn as nn
import torch


class Conv2dSWD(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_radius=2, bias=True):
        super(Conv2dSWD, self).__init__()
        kernel_size_h = 2 * kernel_radius - 1
        self.padding = kernel_radius - 1
        self.convD = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=(kernel_radius, kernel_size_h),
            padding=self.padding, bias=bias)

    def forward(self, input):
        out_D = self.convD(input)
        return out_D[:, :, self.padding:, :]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
