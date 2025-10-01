import torch
import torch.nn as nn
import torch.nn.init
import torch.optim


class WrapPad2d(nn.Module):
    """Create a padding layer that wraps the data

    Arguments:
        padding (int): the size of the padding
    """

    def __init__(self, padding):
        super(WrapPad2d, self).__init__()
        self.padding = padding

    def forward(self, x):
        nx = x.shape[2]
        ny = x.shape[3]
        return x.repeat(1, 1, 3, 3)[:, :, nx - self.padding:2 * nx + self.
            padding, ny - self.padding:2 * ny + self.padding]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'padding': 4}]
