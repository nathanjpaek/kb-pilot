import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=200)
        self.fc3 = nn.Linear(in_features=200, out_features=100)
        self.out = nn.Linear(in_features=100, out_features=10)

    def forward(self, t):
        t = t.view(-1, 28 * 28)
        t = self.fc1(t)
        t = F.relu(t)
        t = self.fc2(t)
        t = F.relu(t)
        t = self.fc3(t)
        t = F.relu(t)
        t = self.out(t)
        return t


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
