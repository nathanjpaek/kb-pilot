import torch
import torch.nn as nn
import torch.nn.functional as F


class QValueFunction(nn.Module):
    """fully connected 200x200 hidden layers"""

    def __init__(self, state_dim, action_dim):
        super(QValueFunction, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 200)
        self.fc2 = nn.Linear(200, 200)
        self.out = nn.Linear(200, 1)

    def forward(self, s, a):
        """return: scalar value"""
        x = torch.cat((s, a), dim=1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        return self.out(x)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4}]
