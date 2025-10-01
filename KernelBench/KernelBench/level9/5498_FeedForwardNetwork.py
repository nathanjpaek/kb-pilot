import math
import torch
import torch.nn as nn


class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 
            0.044715 * torch.pow(x, 3))))


class FeedForwardNetwork(nn.Module):

    def __init__(self, in_dim, hid_dim) ->None:
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.lin2 = nn.Linear(hid_dim, in_dim)
        self.gleu = GELU()
        self.dropout = nn.Dropout()

    def forward(self, inputs):
        hid = self.gleu(self.lin1(inputs))
        return self.lin2(self.dropout(hid))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'hid_dim': 4}]
