import torch
import torch.nn as nn
import torch._C
import torch.serialization
from torch import optim as optim


class ExampleBackbone(nn.Module):

    def __init__(self):
        super(ExampleBackbone, self).__init__()
        self.conv = nn.Conv2d(3, 3, 3)

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        return [self.conv(x)]


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
