import torch
import torch.nn as nn
import torch.autograd


class MatrixConv2dResblock(nn.Module):

    def __init__(self, weight_shape, stride=1, padding=0, with_batchnorm=
        False, act_func='ReLU'):
        super(MatrixConv2dResblock, self).__init__()
        self.conv = nn.Conv2d(weight_shape[3], weight_shape[0],
            weight_shape[1], stride=stride, padding=padding, bias=not
            with_batchnorm)
        if with_batchnorm:
            self.bn = nn.BatchNorm2d(weight_shape[0])
        else:
            self.bn = None
        if act_func is not None:
            self.f = getattr(nn, act_func)()
        else:
            self.f = None

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            y = self.bn(y)
        if self.f is not None:
            y = self.f(y)
        y = torch.add(x, y)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'weight_shape': [4, 4, 4, 4]}]
