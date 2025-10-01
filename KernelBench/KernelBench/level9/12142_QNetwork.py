import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(self, device, action_size, observation_size):
        super(QNetwork, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(np.array((observation_size,)).prod() + np.prod
            ((action_size,)), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.Tensor(x)
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'device': 0, 'action_size': 4, 'observation_size': 4}]
