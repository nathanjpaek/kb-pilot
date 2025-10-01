import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetworkSmall(nn.Module):

    def __init__(self, state_size, action_size, seed):
        """
        Build a fully connected neural network

        state_size (int): State dimension
        action_size (int): Action dimension
        seed (int): random seed
        """
        super(QNetworkSmall, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4, 'seed': 4}]
