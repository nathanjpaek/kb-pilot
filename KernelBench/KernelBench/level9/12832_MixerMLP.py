import torch
import torch.nn as nn


class MixerMLP(nn.Module):
    """Some Information about MixerMLP"""

    def __init__(self, dim, activation='gelu'):
        super(MixerMLP, self).__init__()
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.swapaxes(1, 2)
        x = self.ln(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        x = x.swapaxes(1, 2)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
