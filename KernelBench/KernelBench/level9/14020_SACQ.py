import torch
import torch.nn as nn


class SACQ(nn.Module):

    def __init__(self, n_observations, action_dim, n_hidden):
        super().__init__()
        self.linear = nn.Linear(n_observations, n_hidden)
        self.linear_2 = nn.Linear(action_dim, n_hidden)
        self.linear_q = nn.Linear(n_hidden * 2, n_hidden)
        self.linear_qq = nn.Linear(n_hidden, 1)

    def forward(self, frame, action):
        zf = torch.relu(self.linear(frame))
        za = torch.relu(self.linear_2(action))
        q = torch.relu(self.linear_q(torch.cat([zf, za], dim=1)))
        q = self.linear_qq(q)
        return q


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'n_observations': 4, 'action_dim': 4, 'n_hidden': 4}]
