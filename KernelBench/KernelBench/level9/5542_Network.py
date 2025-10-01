import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def __init__(self, input_size, action_size):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.logits_p = nn.Linear(256, action_size)
        self.v_values = nn.Linear(256, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.logits_p(x), self.v_values(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'action_size': 4}]
