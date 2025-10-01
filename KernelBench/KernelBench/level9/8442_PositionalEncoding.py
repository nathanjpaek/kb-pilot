import torch
import torch.nn as nn
import torch.optim
import torch.nn.init


class PositionalEncoding(nn.Module):

    def __init__(self, emb_size: 'int', spatial_size: 'int'):
        super(PositionalEncoding, self).__init__()
        self.emb_size = emb_size
        self.spatial_size = spatial_size
        self.positions = nn.Parameter(torch.randn(self.emb_size, self.
            spatial_size))

    def forward(self, x):
        x += self.positions
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'emb_size': 4, 'spatial_size': 4}]
