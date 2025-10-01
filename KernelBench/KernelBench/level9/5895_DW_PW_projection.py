import torch
import torch.nn as nn


class DW_PW_projection(nn.Module):

    def __init__(self, c_in, c_out, kernel_size, bias=False, padding_mode=
        'replicate'):
        super(DW_PW_projection, self).__init__()
        self.dw_conv1d = nn.Conv1d(in_channels=c_in, out_channels=c_in,
            kernel_size=kernel_size, padding=int(kernel_size / 2), groups=
            c_in, bias=bias, padding_mode=padding_mode)
        self.pw_conv1d = nn.Conv1d(in_channels=c_in, out_channels=c_out,
            kernel_size=1, padding=0, groups=1, bias=bias, padding_mode=
            padding_mode)

    def forward(self, x):
        x = self.dw_conv1d(x)
        x = self.pw_conv1d(x)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'c_in': 4, 'c_out': 4, 'kernel_size': 4}]
