import torch
import torch.utils.data
from torch import nn


class AffineChannel2d(nn.Module):
    """ A simple channel-wise affine transformation operation """

    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(num_channels))
        self.bias = nn.Parameter(torch.Tensor(num_channels))
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, x):
        return x * self.weight.view(1, self.num_channels, 1, 1
            ) + self.bias.view(1, self.num_channels, 1, 1)

    def __repr__(self):
        return 'AffineChannel2d(num_features={}, eps={})'.format(self.
            num_channels, self.eps)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_channels': 4}]
