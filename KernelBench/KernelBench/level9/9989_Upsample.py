import torch
import torch.nn as nn
import torch._utils
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode='linear'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'scale_factor': 1.0}]
