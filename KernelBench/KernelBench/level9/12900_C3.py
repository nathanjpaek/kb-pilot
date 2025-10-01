import torch
import torch.nn as nn
from collections import OrderedDict


class C3(nn.Module):

    def __init__(self):
        super(C3, self).__init__()
        self.c3 = nn.Sequential(OrderedDict([('c3', nn.Conv2d(16, 120,
            kernel_size=(5, 5))), ('relu3', nn.ReLU())]))

    def forward(self, img):
        output = self.c3(img)
        return output


def get_inputs():
    return [torch.rand([4, 16, 64, 64])]


def get_init_inputs():
    return [[], {}]
