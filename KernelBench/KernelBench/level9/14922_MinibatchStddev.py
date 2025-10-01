import torch
from torch import nn


def Tstdeps(val):
    return torch.sqrt(((val - val.mean()) ** 2).mean() + 1e-08)


class MinibatchStddev(nn.Module):

    def __init__(self):
        super(MinibatchStddev, self).__init__()
        self.eps = 1.0

    def forward(self, x):
        stddev_mean = Tstdeps(x)
        new_channel = stddev_mean.expand(x.size(0), 1, x.size(2), x.size(3))
        h = torch.cat((x, new_channel), dim=1)
        return h


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
