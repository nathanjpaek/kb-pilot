import torch
import torch.nn as nn
import torch.nn.init


class DownsampleA(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        return torch.cat((self.avg(x), x.mul(0)), 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nIn': 4, 'nOut': 4, 'stride': 1}]
