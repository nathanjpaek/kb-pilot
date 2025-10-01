from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
from torchvision.transforms import *


class _Transition(nn.Module):

    def __init__(self, in_channels, args):
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'args': _mock_config()}]
