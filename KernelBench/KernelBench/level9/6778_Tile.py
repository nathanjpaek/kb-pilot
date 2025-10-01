import torch
import torch.nn as nn


class Tile(nn.Module):

    def __init__(self, max_size, dim):
        super(Tile, self).__init__()
        self.max_size = max_size
        self.dim = dim

    def forward(self, input):
        return input.repeat(*[(self.max_size if x == self.dim else 1) for x in
            range(len(input.shape))])


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'max_size': 4, 'dim': 4}]
