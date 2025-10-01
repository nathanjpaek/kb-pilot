import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample(nn.Module):

    def __init__(self, strides=(2, 2), **kwargs):
        super(Downsample, self).__init__()
        if isinstance(strides, int):
            strides = strides, strides
        self.strides = strides

    def forward(self, x):
        shape = -(-x.size()[2] // self.strides[0]), -(-x.size()[3] // self.
            strides[1])
        x = F.interpolate(x, size=shape, mode='nearest')
        return x

    def extra_repr(self):
        return 'strides=%s' % repr(self.strides)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
