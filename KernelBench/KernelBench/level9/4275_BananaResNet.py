import torch
import torch.nn as nn
import torch.nn.functional as F


class BananaResNet(nn.Module):

    def __init__(self, state_size, action_size):
        super(BananaResNet, self).__init__()
        self.blk1fc1 = nn.Linear(state_size, 128)
        self.blk1fc2 = nn.Linear(128, 128)
        self.blk1fc3 = nn.Linear(128, 64)
        self.blk2fc1 = nn.Linear(64, 64)
        self.blk2fc2 = nn.Linear(64, 64)
        self.blk2fc3 = nn.Linear(64, 32)
        self.outfc = nn.Linear(32, action_size)

    def forward(self, state):
        skip = F.relu(self.blk1fc1(state))
        x = F.relu(self.blk1fc2(skip))
        x = x + skip
        skip = F.relu(self.blk1fc3(x))
        x = F.relu(self.blk2fc1(skip))
        skip = x + skip
        x = F.relu(self.blk2fc2(x))
        skip = x + skip
        x = F.relu(self.blk2fc3(skip))
        x = self.outfc(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4}]
