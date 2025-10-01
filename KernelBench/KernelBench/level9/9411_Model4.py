import torch
from torch import nn


class Model4(nn.Module):

    def __init__(self, input_dim, output_dim, hidden=64):
        super(Model4, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden, hidden)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
