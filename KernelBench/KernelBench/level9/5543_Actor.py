import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):

    def __init__(self, input_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.output = nn.Linear(200, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = torch.sigmoid(self.output(x))
        return action_prob


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'action_size': 4}]
