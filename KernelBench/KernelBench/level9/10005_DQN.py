import torch
import torch.nn.functional as F
from torch import nn


class DQN(nn.Module):
    """DQN network, three full connection layers
    """

    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(16, 2)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
