import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):

    def __init__(self, num_inputs, num_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 100)
        self.action_head = nn.Linear(100, num_actions)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'num_inputs': 4, 'num_actions': 4}]
