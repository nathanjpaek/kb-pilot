import torch
from torch import nn
import torch.nn.functional as F


class Critic(nn.Module):
    """Critic model

        Parameters:
              args (object): Parameter class

    """

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        l1 = 400
        l2 = 300
        self.q1f1 = nn.Linear(state_dim + action_dim, l1)
        self.q1ln1 = nn.LayerNorm(l1)
        self.q1f2 = nn.Linear(l1, l2)
        self.q1ln2 = nn.LayerNorm(l2)
        self.q1out = nn.Linear(l2, 1)
        self.q2f1 = nn.Linear(state_dim + action_dim, l1)
        self.q2ln1 = nn.LayerNorm(l1)
        self.q2f2 = nn.Linear(l1, l2)
        self.q2ln2 = nn.LayerNorm(l2)
        self.q2out = nn.Linear(l2, 1)
        self.vf1 = nn.Linear(state_dim, l1)
        self.vln1 = nn.LayerNorm(l1)
        self.vf2 = nn.Linear(l1, l2)
        self.vln2 = nn.LayerNorm(l2)
        self.vout = nn.Linear(l2, 1)

    def forward(self, obs, action):
        """Method to forward propagate through the critic's graph

             Parameters:
                   input (tensor): states
                   input (tensor): actions

             Returns:
                   Q1 (tensor): Qval 1
                   Q2 (tensor): Qval 2
                   V (tensor): Value



         """
        state = torch.cat([obs, action], 1)
        q1 = F.elu(self.q1f1(state))
        q1 = self.q1ln1(q1)
        q1 = F.elu(self.q1f2(q1))
        q1 = self.q1ln2(q1)
        q1 = self.q1out(q1)
        q2 = F.elu(self.q2f1(state))
        q2 = self.q2ln1(q2)
        q2 = F.elu(self.q2f2(q2))
        q2 = self.q2ln2(q2)
        q2 = self.q2out(q2)
        v = F.elu(self.vf1(obs))
        v = self.vln1(v)
        v = F.elu(self.vf2(v))
        v = self.vln2(v)
        v = self.vout(v)
        return q1, q2, v


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4}]
