import torch
import torch as t
import torch.nn as nn


class DoubleInputNet(nn.Module):

    def __init__(self, firstinsize, secondinsize, outsize, activation=lambda
        x: x):
        super().__init__()
        self.firstinsize = firstinsize
        self.secondinsize = secondinsize
        self.outsize = outsize
        self.activation = activation
        self.fc1_1 = nn.Linear(firstinsize, 64)
        self.fc1_2 = nn.Linear(secondinsize, 64)
        self.fc2 = nn.Linear(128, 64)
        self.head = nn.Linear(64, self.outsize)

    def forward(self, firstin, secondin):
        x1 = nn.functional.relu(self.fc1_1(firstin))
        x2 = nn.functional.relu(self.fc1_2(secondin))
        x = t.cat([x1, x2], dim=1)
        x = nn.functional.relu(self.fc2(x))
        return self.activation(self.head(x))


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'firstinsize': 4, 'secondinsize': 4, 'outsize': 4}]
