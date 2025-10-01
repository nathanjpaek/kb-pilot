import torch
import torch.nn as nn
from collections import OrderedDict


class VGG19Decoder1(nn.Module):

    def __init__(self):
        super(VGG19Decoder1, self).__init__()
        self.blocks = OrderedDict([('pad1_1', nn.ReflectionPad2d(1)), (
            'conv1_1', nn.Conv2d(64, 3, 3, 1, 0))])
        self.seq = nn.Sequential(self.blocks)

    def forward(self, x, targets=None):
        return self.seq(x)


def get_inputs():
    return [torch.rand([4, 64, 4, 4])]


def get_init_inputs():
    return [[], {}]
