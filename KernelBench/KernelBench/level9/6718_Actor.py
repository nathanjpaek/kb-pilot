import torch
import torch.nn.functional as F
import torch.nn as nn


class Actor(nn.Module):

    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        nn.init.normal_(self.linear1.weight, 0.0, 0.02)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.linear2.weight, 0.0, 0.02)
        self.mu = nn.Linear(hidden_size, num_outputs)
        torch.nn.init.uniform_(self.mu.weight, a=-0.003, b=0.003)

    def forward(self, inputs):
        x = inputs
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        mu = F.tanh(self.mu(x))
        return mu


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'num_inputs': 4, 'action_space': torch.
        rand([4, 4])}]
