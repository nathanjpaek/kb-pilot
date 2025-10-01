import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):

    def __init__(self, dim):
        super(Block, self).__init__()
        self.dim = dim
        self.layer_norm = nn.LayerNorm(self.dim)
        self.conv = nn.Conv1d(self.dim, self.dim, kernel_size=3, padding=1)

    def forward(self, x):
        x_orig = x
        x = F.relu(x)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x + x_orig


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
