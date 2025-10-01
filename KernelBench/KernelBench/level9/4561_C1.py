import torch
import torch.nn as nn
from collections import OrderedDict


class C1(nn.Module):

    def __init__(self):
        super(C1, self).__init__()
        self.c1 = nn.Sequential(OrderedDict([('c1', nn.Conv2d(1, 6,
            kernel_size=(5, 5))), ('relu1', nn.ReLU()), ('s1', nn.MaxPool2d
            (kernel_size=(2, 2), stride=2))]))

    def forward(self, img):
        output = self.c1(img)
        return output


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
