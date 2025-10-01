import torch
import torch.nn as nn
import torch.utils.data
import torch.utils
from matplotlib import cm as cm
from torch.nn.parallel import *
from torchvision.models import *
from torchvision.datasets import *


def get_norm_layer(norm, C):
    if norm in [None, '', 'none']:
        norm_layer = nn.Identity()
    elif norm.startswith('bn'):
        norm_layer = nn.BatchNorm2d(C, track_running_stats=norm.find(
            'track') >= 0)
    else:
        raise NotImplementedError(norm)
    return norm_layer


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, norm='bn', stride=2):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.stride = stride
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=stride, padding
            =0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=stride, padding
            =0, bias=False)
        self.bn = get_norm_layer(norm, C_out)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:] if 
            self.stride > 1 else x)], dim=1)
        out = self.bn(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'C_in': 4, 'C_out': 4}]
