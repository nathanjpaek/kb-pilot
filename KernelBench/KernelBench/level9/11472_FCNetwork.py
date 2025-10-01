import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNetwork(nn.Module):

    def __init__(self, state_size, action_size, output_gate=None):
        super(FCNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)
        self.output_gate = output_gate

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.output_gate is not None:
            x = self.output_gate(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4}]
