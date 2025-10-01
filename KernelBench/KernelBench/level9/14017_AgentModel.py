import torch
import torch.nn as nn


class AgentModel(nn.Module):
    """The model that computes one score per action"""

    def __init__(self, n_observations, n_actions, n_hidden):
        super().__init__()
        self.linear = nn.Linear(n_observations, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_actions)

    def forward(self, frame):
        z = torch.tanh(self.linear(frame))
        score_actions = self.linear2(z)
        probabilities_actions = torch.softmax(score_actions, dim=-1)
        return probabilities_actions


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_observations': 4, 'n_actions': 4, 'n_hidden': 4}]
