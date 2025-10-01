import torch
from torch import nn
from torch.nn import functional as F


class Value(nn.Module):

    def __init__(self, state_size, fcs1_units=400, fc2_units=300):
        super(Value, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(5 * 5 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, state):
        x = state.view(-1, 1, 32, 32)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 50)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 32, 32])]


def get_init_inputs():
    return [[], {'state_size': 4}]
