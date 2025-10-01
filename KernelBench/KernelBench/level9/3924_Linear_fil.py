import torch
import torch.nn as nn


class Linear_fil(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(Linear_fil, self).__init__()
        self.lin_1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.lin_2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.lin_1(x)
        x = self.act(x)
        x = self.lin_2(x)
        x = self.sigmoid(x).squeeze()
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'hidden_dim': 4}]
