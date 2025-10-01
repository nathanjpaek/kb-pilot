import torch
import torch.nn.functional as F
import torch.nn as nn


class policy_net(nn.Module):

    def __init__(self, n_states, n_actions, n_hidden=128):
        super(policy_net, self).__init__()
        self.affine1 = nn.Linear(n_states, n_hidden)
        self.affine2 = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_states': 4, 'n_actions': 4}]
