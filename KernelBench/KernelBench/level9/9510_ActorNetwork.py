import torch
import torch.nn.functional as F
import torch.nn as nn


class ActorNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed):
        super(ActorNetwork, self).__init__()
        torch.manual_seed(seed)
        hidden1 = 64
        hidden2 = 64
        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return torch.tanh(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4, 'seed': 4}]
