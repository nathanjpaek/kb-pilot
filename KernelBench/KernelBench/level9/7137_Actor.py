import torch
import torch.nn.functional as F
import torch.nn as nn


class Actor(torch.nn.Module):

    def __init__(self, numObs, numActions):
        super(Actor, self).__init__()
        self.actor_input = nn.Linear(numObs, 32)
        self.actor_fc1 = nn.Linear(32, 32)
        self.actor_output = nn.Linear(32, numActions)

    def forward(self, x):
        x = F.relu(self.actor_input(x))
        x = F.relu(self.actor_fc1(x))
        logits = self.actor_output(x)
        return logits


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'numObs': 4, 'numActions': 4}]
