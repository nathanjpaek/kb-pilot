import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    """Agent network"""

    def __init__(self, in_size, out_size):
        super().__init__()
        self.fc1 = nn.Linear(in_size, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 50)
        self.out = nn.Linear(50, out_size)

    def forward(self, t):
        if len(t.shape) == 3:
            t = t.unsqueeze(0)
        t = self.fc1(t)
        t = F.relu(t)
        t = self.fc2(t)
        t = F.relu(t)
        t = self.fc3(t)
        t = F.relu(t)
        return self.out(t)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4, 'out_size': 4}]
