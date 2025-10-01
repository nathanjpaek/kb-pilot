import torch
import torch.nn as nn
import torch.nn.init


class maxPool23DUinit(nn.Module):

    def __init__(self, kernel_size, stride, padding=1, dilation=1, nd=2):
        super(maxPool23DUinit, self).__init__()
        assert nd == 1 or nd == 2 or nd == 3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        if nd == 2:
            self.pool1 = nn.MaxPool2d(kernel_size=kernel_size, stride=
                stride, padding=padding, dilation=dilation)
        elif nd == 3:
            self.pool1 = nn.MaxPool3d(kernel_size=kernel_size, stride=
                stride, padding=padding, dilation=dilation)
        else:
            self.pool1 = nn.MaxPool1d(kernel_size=kernel_size, stride=
                stride, padding=padding, dilation=dilation)

    def forward(self, x):
        return self.pool1(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4, 'stride': 1}]
