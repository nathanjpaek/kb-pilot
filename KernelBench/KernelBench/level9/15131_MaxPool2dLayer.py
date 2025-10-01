import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxPool2dLayer(nn.Module):

    def forward(self, tensor, kernel_size=(3, 3), stride=(1, 1), padding=0,
        ceil_mode=False):
        return F.max_pool2d(tensor, kernel_size, stride=stride, padding=
            padding, ceil_mode=ceil_mode)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
