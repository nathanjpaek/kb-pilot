import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """ 全连接网络"""

    def __init__(self, state_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, 36)
        self.fc2 = nn.Linear(36, 36)
        self.fc3 = nn.Linear(36, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4}]
