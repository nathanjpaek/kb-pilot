import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data


class Embedding(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Embedding, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = self.tanh(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
