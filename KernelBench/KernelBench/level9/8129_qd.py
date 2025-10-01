import torch
import torch.nn as nn


class qd(nn.Module):

    def __init__(self, d_dim, zd_dim):
        super(qd, self).__init__()
        self.fc1 = nn.Linear(zd_dim, d_dim)
        self.activation = nn.LeakyReLU(inplace=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zd):
        h = self.activation(zd)
        loc_d = self.fc1(h)
        return loc_d


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_dim': 4, 'zd_dim': 4}]
