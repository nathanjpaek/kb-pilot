import torch
import torch.utils.data
from torch.utils import data as data
import torch.nn as nn
from torch.nn import init as init
from torchvision.models import vgg as vgg
from torch import autograd as autograd


class FakeReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class FakeReLUM(nn.Module):

    def forward(self, x):
        return FakeReLU.apply(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
