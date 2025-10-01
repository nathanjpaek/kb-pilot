import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


class Parseval_Conv2d(nn.Conv2d):

    def forward(self, input):
        new_weight = self.weight / np.sqrt(2 * self.kernel_size[0] * self.
            kernel_size[1] + 1)
        return F.conv2d(input, new_weight, self.bias, self.stride, self.
            padding, self.dilation, self.groups)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
