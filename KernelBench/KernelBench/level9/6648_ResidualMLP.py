import torch
import torch.nn as nn


class ResidualMLP(nn.Module):

    def __init__(self, input_dim, target_dim, hidden_dim=64):
        super(ResidualMLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, target_dim)

    def forward(self, h):
        h = self.linear1(h).relu()
        h = h + self.linear2(h).relu()
        h = h + self.linear3(h).relu()
        return self.linear4(h)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'target_dim': 4}]
