import torch
from typing import *
import torch.nn as nn
import torch.nn.functional as F


class LeNet_300_100(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, 784)))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
