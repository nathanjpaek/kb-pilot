import torch
from torch import nn
import torch.nn.functional as F


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.fc4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        x1 = F.relu(self.fc1(sa))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)
        x2 = F.relu(self.fc4(sa))
        x2 = F.relu(self.fc5(x2))
        x2 = self.fc6(x2)
        return x1, x2


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'num_inputs': 4, 'num_actions': 4, 'hidden_dim': 4}]
