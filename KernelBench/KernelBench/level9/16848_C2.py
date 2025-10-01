import torch
import torch.nn as nn
from collections import OrderedDict


class C2(nn.Module):

    def __init__(self):
        super(C2, self).__init__()
        self.c2 = nn.Sequential(OrderedDict([('c2', nn.Conv2d(6, 16,
            kernel_size=(5, 5))), ('relu2', nn.ReLU()), ('s2', nn.MaxPool2d
            (kernel_size=(2, 2), stride=2))]))

    def forward(self, img):
        output = self.c2(img)
        return output


def get_inputs():
    return [torch.rand([4, 6, 64, 64])]


def get_init_inputs():
    return [[], {}]
