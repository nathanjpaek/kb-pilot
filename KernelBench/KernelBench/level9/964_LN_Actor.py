import torch
import torch.nn as nn
import torch.nn.functional as F


class LN_Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action, hidden_size1,
        hidden_size2):
        super(LN_Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_size1)
        self.ln1 = nn.LayerNorm(hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.ln2 = nn.LayerNorm(hidden_size2)
        self.l3 = nn.Linear(hidden_size2, action_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.ln1(x)
        x = F.relu(self.l2(x))
        x = self.ln2(x)
        x = torch.tanh(self.l3(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4, 'max_action': 4,
        'hidden_size1': 4, 'hidden_size2': 4}]
