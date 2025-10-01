import torch
import numpy as np
import torch.nn as nn


class LinearConvExpansion(nn.Module):
    """expansion 1D -> 2D"""

    def __init__(self, d_features, n_channel, n_depth):
        super().__init__()
        self.d_features = d_features
        self.n_channel = n_channel
        self.n_depth = n_depth
        self.d_hid = int(np.round(np.sqrt(d_features)))
        self.linear = nn.Linear(d_features, self.d_hid * d_features)
        self.conv = nn.Conv1d(self.d_hid, n_channel * n_depth, kernel_size=1)

    def forward(self, x):
        """
            Arguments:
                x {Tensor, shape [batch, d_features]} -- input

            Returns:
                x {Tensor, shape [batch, n_channel * n_depth, d_features]} -- output
        """
        x = self.linear(x).view(-1, self.d_hid, self.d_features)
        x = self.conv(x)
        return x

    def initialize_param(self, init, *args):
        init(self.linear.weight, *args)
        init(self.conv.weight, *args)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_features': 4, 'n_channel': 4, 'n_depth': 1}]
