import torch
import torch.nn as nn
import torch.nn.functional as F


class ATT(nn.Module):

    def __init__(self, din):
        super(ATT, self).__init__()
        self.fc1 = nn.Linear(din, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = F.sigmoid(self.fc3(y))
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'din': 4}]
