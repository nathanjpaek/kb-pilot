import torch
import torch.nn as nn


class MediatorNet(nn.Module):

    def __init__(self, input_dim):
        super(MediatorNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim * 3)
        self.fc2 = nn.Linear(input_dim * 3, input_dim * 3)
        self.fc_last = nn.Linear(input_dim * 3, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc_last(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
