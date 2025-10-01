import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


class Upsample(nn.Module):

    def __init__(self, scale_factor=2, size=None):
        super(Upsample, self).__init__()
        self.upsample = F.upsample_nearest
        self.size = size
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.upsample(x, size=self.size, scale_factor=self.scale_factor)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
