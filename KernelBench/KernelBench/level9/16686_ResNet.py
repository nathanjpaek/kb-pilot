import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):

    def __init__(self, n_in, n_out):
        super(ResNet, self).__init__()
        self.fc1 = nn.Linear(n_in, n_out)
        self.fc2 = nn.Linear(n_in, n_out)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1) + x)
        return h2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_in': 4, 'n_out': 4}]
