import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast


class LowRankPositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1_u = nn.Linear(d_in, int(d_in / 4), bias=False)
        self.w_1_v = nn.Linear(int(d_in / 4), d_hid)
        self.w_2_u = nn.Linear(d_hid, int(d_in / 4), bias=False)
        self.w_2_v = nn.Linear(int(d_in / 4), d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-06)
        self.dropout = nn.Dropout(dropout)

    @autocast()
    def forward(self, x):
        residual = x
        x = self.w_2_v(self.w_2_u(F.relu(self.w_1_v(self.w_1_u(x)))))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_in': 4, 'd_hid': 4}]
