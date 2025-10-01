import torch
import torch.nn as nn
import torch.nn.functional as F


class FC_Q(nn.Module):

    def __init__(self, state_dim, num_actions, num_nodes=128):
        super(FC_Q, self).__init__()
        self.q1 = nn.Linear(state_dim, num_nodes)
        self.q2 = nn.Linear(num_nodes, num_nodes)
        self.q3 = nn.Linear(num_nodes, num_actions)
        self.i1 = nn.Linear(state_dim, num_nodes)
        self.i2 = nn.Linear(num_nodes, num_nodes)
        self.i3 = nn.Linear(num_nodes, num_actions)

    def forward(self, state):
        q = F.relu(self.q1(state))
        q = F.relu(self.q2(q))
        i = F.relu(self.i1(state))
        i = F.relu(self.i2(i))
        i = F.relu(self.i3(i))
        return self.q3(q), F.log_softmax(i, dim=1), i


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'num_actions': 4}]
