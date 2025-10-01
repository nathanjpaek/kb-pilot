import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetDownsampleA(nn.Module):

    def __init__(self, planes):
        super(ResNetDownsampleA, self).__init__()
        self._planes = planes

    def forward(self, x):
        return F.pad(input=x[:, :, ::2, ::2], pad=(0, 0, 0, 0, self._planes //
            4, self._planes // 4), mode='constant', value=0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'planes': 4}]
