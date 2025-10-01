import torch
from torch import nn as nn
from torch.nn import init as init
from torchvision.models import vgg as vgg
import torch.utils.data
from torch.utils import data as data
from torch import autograd as autograd


class SoftGate(nn.Module):
    COEFF = 12.0

    def __init__(self):
        super(SoftGate, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x).mul(self.COEFF)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
