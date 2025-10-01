import torch
import torch.nn.functional as F
import torch.nn as nn


class ac_net(nn.Module):

    def __init__(self, n_states, n_actions, n_hidden=32):
        super(ac_net, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.action_head = nn.Linear(n_hidden, n_actions)
        self.value_head = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_score = self.action_head(x)
        state_value = self.value_head(x)
        return F.softmax(action_score, dim=-1), state_value


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_states': 4, 'n_actions': 4}]
