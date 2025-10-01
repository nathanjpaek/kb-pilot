import torch
import numpy as np
import torch.nn as nn
from typing import Tuple


def to_sate_tensor(s, device):
    """ converts a numpy array to a Tensor suitable for passing through DQNs """
    return torch.from_numpy(s)


class ConvEncoder(nn.Module):

    def __init__(self, state_shape: 'Tuple', device=None):
        super(ConvEncoder, self).__init__()
        in_channels = state_shape[0]
        nc = 32
        self.conv1 = nn.Conv2d(in_channels, nc, (8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(nc, 2 * nc, (4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(2 * nc, 2 * nc, (3, 3), stride=(1, 1))
        self.relu = nn.ReLU()
        self.state_shape = state_shape

    def forward(self, s) ->torch.Tensor:
        if isinstance(s, np.ndarray):
            s = to_sate_tensor(s, self.device)
        a = self.relu(self.conv1(s))
        a = self.relu(self.conv2(a))
        a = self.relu(self.conv3(a))
        return torch.flatten(a, 1)

    @property
    def output_shape(self):
        """ the output shape of this CNN encoder.
        :return: tuple of output shape
        """
        return 64, 12, 12

    def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
        """ Utility function for computing output of convolutions
        takes a tuple of (h,w) and returns a tuple of (h,w)
        """
        from math import floor
        if type(kernel_size) is not tuple:
            kernel_size = kernel_size, kernel_size
        h = floor((h_w[0] + 2 * pad - dilation * (kernel_size[0] - 1) - 1) /
            stride + 1)
        w = floor((h_w[1] + 2 * pad - dilation * (kernel_size[1] - 1) - 1) /
            stride + 1)
        return h, w


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'state_shape': [4, 4]}]
