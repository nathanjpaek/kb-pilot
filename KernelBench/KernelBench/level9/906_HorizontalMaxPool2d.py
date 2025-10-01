import torch
import torch.nn as nn


class HorizontalMaxPool2d(nn.Module):

    def __init__(self):
        super(HorizontalMaxPool2d, self).__init__()

    def forward(self, x):
        inp_size = x.size()
        return nn.functional.max_pool2d(input=x, kernel_size=(1, inp_size[3]))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
