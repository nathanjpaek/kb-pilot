import torch
import torch.nn as nn
import torch.cuda
import torch.distributed


class SoftmaxWithTemperature(nn.Module):

    def __init__(self, dim=0, alpha=1.0):
        super(SoftmaxWithTemperature, self).__init__()
        self._softmax = nn.Softmax(dim)
        self._alpha = alpha

    def forward(self, x):
        return self._softmax(self._alpha * x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
