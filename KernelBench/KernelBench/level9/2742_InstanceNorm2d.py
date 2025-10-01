import torch
from torch import nn as nn
from torch.nn import init as init
from torchvision.models import vgg as vgg
import torch.utils.data
from torch.utils import data as data
from torch import autograd as autograd


class InstanceNorm2d(nn.Module):

    def __init__(self, epsilon=1e-08, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def forward(self, x):
        tmp = torch.mul(x, x)
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
