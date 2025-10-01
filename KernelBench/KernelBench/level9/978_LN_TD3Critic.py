import torch
import torch.nn as nn
import torch.nn.functional as F


class LN_TD3Critic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_size1, hidden_size2):
        super(LN_TD3Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_size1)
        self.ln1 = nn.LayerNorm(hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.ln2 = nn.LayerNorm(hidden_size2)
        self.l3 = nn.Linear(hidden_size2, 1)
        self.l4 = nn.Linear(state_dim + action_dim, hidden_size1)
        self.ln4 = nn.LayerNorm(hidden_size1)
        self.l5 = nn.Linear(hidden_size1, hidden_size2)
        self.ln5 = nn.LayerNorm(hidden_size2)
        self.l6 = nn.Linear(hidden_size2, 1)

    def forward(self, inputs, actions):
        xu = torch.cat([inputs, actions], 1)
        x1 = F.relu(self.l1(xu))
        x1 = self.ln1(x1)
        x1 = F.relu(self.l2(x1))
        x1 = self.ln2(x1)
        x1 = self.l3(x1)
        x2 = F.relu(self.l4(xu))
        x2 = self.ln4(x2)
        x2 = F.relu(self.l5(x2))
        x2 = self.ln5(x2)
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, inputs, actions):
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
