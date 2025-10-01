from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class EnchanceReLU(nn.ReLU):

    def __init__(self, args):
        super(EnchanceReLU, self).__init__(inplace=True)
        self.shift = getattr(args, 'fm_boundary', 0.25)

    def forward(self, x):
        x = x + self.shift
        x = super(EnchanceReLU, self).forward(x)
        x = x - self.shift
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'args': _mock_config(fm_boundary=4)}]
