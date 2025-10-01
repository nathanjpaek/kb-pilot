import torch
import torch.nn as nn
import torch.autograd


class MatrixConv2dMultiResblock(nn.Module):

    def __init__(self, weight_shape, stride=1, padding=0, with_batchnorm=
        False, act_func='ReLU'):
        super(MatrixConv2dMultiResblock, self).__init__()
        self.conv1 = nn.Conv2d(weight_shape[3], weight_shape[0],
            weight_shape[1], stride=stride, padding=padding, bias=not
            with_batchnorm)
        if with_batchnorm:
            self.bn1 = nn.BatchNorm2d(weight_shape[0])
        else:
            self.bn1 = None
        if act_func is not None:
            self.f1 = getattr(nn, act_func)()
        else:
            self.f1 = None
        self.conv2 = nn.Conv2d(weight_shape[3], weight_shape[0],
            weight_shape[1], stride=stride, padding=padding, bias=not
            with_batchnorm)
        if with_batchnorm:
            self.bn2 = nn.BatchNorm2d(weight_shape[0])
        else:
            self.bn2 = None
        if act_func is not None:
            self.f2 = getattr(nn, act_func)()
        else:
            self.f2 = None

    def forward(self, x):
        y = self.conv1(x)
        if self.bn1 is not None:
            y = self.bn1(y)
        if self.f1 is not None:
            y = self.f1(y)
        y = torch.add(x, y)
        x = y
        y = self.conv2(y)
        if self.bn2 is not None:
            y = self.bn2(y)
        if self.f2 is not None:
            y = self.f2(y)
        y = torch.add(x, y)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'weight_shape': [4, 4, 4, 4]}]
