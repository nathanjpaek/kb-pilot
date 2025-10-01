import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        out = max(8, in_dim * 2)
        self.input = nn.Linear(in_dim, out)
        self.fc = nn.Linear(out, out)
        self.fc2 = nn.Linear(out, out)
        self.output = nn.Linear(out, out_dim)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = x.float()
        x = self.relu(self.input(x))
        x = self.relu(self.fc(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.output(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
