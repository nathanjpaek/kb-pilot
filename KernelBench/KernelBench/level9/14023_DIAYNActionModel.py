import torch
import torch.nn as nn


class DIAYNActionModel(nn.Module):
    """The model that computes one score per action"""

    def __init__(self, n_observations, n_actions, n_hidden, n_policies):
        super().__init__()
        self.linear = nn.Linear(n_observations, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_actions * n_policies)
        self.n_policies = n_policies
        self.n_actions = n_actions

    def forward(self, frame, idx_policy):
        z = torch.tanh(self.linear(frame))
        score_actions = self.linear2(z)
        s = score_actions.size()
        score_actions = score_actions.reshape(s[0], self.n_policies, self.
            n_actions)
        score_actions = score_actions[torch.arange(s[0]), idx_policy]
        probabilities_actions = torch.softmax(score_actions, dim=-1)
        return probabilities_actions


def get_inputs():
    return [torch.rand([4, 4]), torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {'n_observations': 4, 'n_actions': 4, 'n_hidden': 4,
        'n_policies': 4}]
