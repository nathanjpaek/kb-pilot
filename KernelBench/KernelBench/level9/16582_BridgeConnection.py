import torch
import torch.nn as nn
from torch.utils import tensorboard as tensorboard


class BridgeConnection(nn.Module):

    def __init__(self, in_dim, out_dim, dout_p):
        super(BridgeConnection, self).__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dout_p)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        x = self.dropout(x)
        return self.activation(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4, 'dout_p': 0.5}]
