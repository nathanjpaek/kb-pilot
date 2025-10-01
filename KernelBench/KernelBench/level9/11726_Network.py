import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def __init__(self, inS, outS):
        super().__init__()
        self.input_size = inS
        self.fc1 = nn.Linear(in_features=inS, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=outS)

    def forward(self, t):
        t = t.reshape(-1, self.input_size)
        t = self.fc1(t)
        t = F.relu(t)
        t = self.fc2(t)
        t = F.relu(t)
        t = self.out(t)
        t = F.softmax(t, dim=1)
        return t


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inS': 4, 'outS': 4}]
