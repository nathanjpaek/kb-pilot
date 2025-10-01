import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):

    def __init__(self, input_size, num_actions):
        super(Policy, self).__init__()
        self.affines = nn.Linear(input_size, 100)
        self.action_head = nn.Linear(100, num_actions)
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        action_scores = F.softmax(self.action_head(F.relu(self.affines(x))),
            dim=-1)
        return action_scores


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'num_actions': 4}]
