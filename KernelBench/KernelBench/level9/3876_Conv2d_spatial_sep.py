import torch
import torch.nn as nn


class Conv2d_spatial_sep(nn.Module):

    def __init__(self, nin, nout):
        super(Conv2d_spatial_sep, self).__init__()
        self.conv1 = nn.Conv2d(nin, 1, kernel_size=(1, 3), groups=1, padding=0)
        self.conv2 = nn.Conv2d(1, nout, kernel_size=(3, 1), groups=1, padding=1
            )

    def forward(self, x):
        return self.conv2(self.conv1(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nin': 4, 'nout': 4}]
