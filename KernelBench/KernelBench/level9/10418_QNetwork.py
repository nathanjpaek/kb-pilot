import torch
import torch.nn.functional as F
import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, hidden_layer1=64,
        hidden_layer2=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.fc3 = nn.Linear(hidden_layer2, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.relu(self.fc3(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4}]
