import torch
import torch.nn as nn


class AgentReinforce(nn.Module):

    def __init__(self, state_shape, n_actions):
        super().__init__()
        self.name = 'reinforce'
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.hidden1 = nn.Linear(self.state_shape, 100)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(100, 100)
        self.act2 = nn.ReLU()
        self.out1 = nn.Linear(100, self.n_actions)

    def forward(self, state_t):
        h = self.act1(self.hidden1(state_t))
        h = self.act2(self.hidden2(h))
        logits = self.out1(h)
        return logits


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_shape': 4, 'n_actions': 4}]
