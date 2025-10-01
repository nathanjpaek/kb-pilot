import torch
import torch.nn as nn


class ACNetwork(nn.Module):

    def __init__(self, num_actions, num_states):
        super(ACNetwork, self).__init__()
        self.fc1 = nn.Linear(num_states, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.action = nn.Linear(512, num_actions)
        self.softmax = nn.Softmax(1)
        self.value = nn.Linear(512, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        pi = self.softmax(self.action(x))
        val = self.value(x)
        return pi, val


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_actions': 4, 'num_states': 4}]
