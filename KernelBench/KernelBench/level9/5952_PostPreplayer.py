import torch
from torch import nn
import torch.nn.functional as F


class PostPreplayer(nn.Module):

    def __init__(self, dim, out_dim, num_nodes, seq_l, dropout):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm((dim, num_nodes, seq_l))
        self.end_conv_1 = nn.Conv2d(in_channels=dim, out_channels=out_dim **
            2, kernel_size=(1, seq_l))
        self.end_conv_2 = nn.Conv2d(in_channels=out_dim ** 2, out_channels=
            out_dim, kernel_size=(1, 1))
        self.dim = dim
        self.seq_l = seq_l
        self.num_nodes = num_nodes
        self.dropout = dropout

    def forward(self, x):
        h = self.norm1(x)
        h = F.relu(self.end_conv_1(h))
        h = self.end_conv_2(h)
        return h


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'out_dim': 4, 'num_nodes': 4, 'seq_l': 4,
        'dropout': 0.5}]
