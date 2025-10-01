import torch
import torch.nn as nn
import torch.nn.functional as F


class LN_DDPGCritic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_size1, hidden_size2):
        super(LN_DDPGCritic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_size1)
        self.ln1 = nn.LayerNorm(hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.ln2 = nn.LayerNorm(hidden_size2)
        self.l3 = nn.Linear(hidden_size2, 1)

    def forward(self, inputs, actions):
        xu = torch.cat([inputs, actions], 1)
        x1 = F.relu(self.l1(xu))
        x1 = self.ln1(x1)
        x1 = F.relu(self.l2(x1))
        x1 = self.ln2(x1)
        x1 = self.l3(x1)
        return x1


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4, 'hidden_size1': 4,
        'hidden_size2': 4}]
