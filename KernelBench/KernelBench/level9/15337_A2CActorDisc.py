import torch
from torch.distributions import Categorical
import torch as t
import torch.nn as nn


class A2CActorDisc(nn.Module):

    def __init__(self, state_dim, action_num):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_num)

    def forward(self, state, action=None):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        probs = t.softmax(self.fc3(a), dim=1)
        dist = Categorical(probs=probs)
        act = action if action is not None else dist.sample()
        act_entropy = dist.entropy()
        act_log_prob = dist.log_prob(act.flatten())
        return act, act_log_prob, act_entropy


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_num': 4}]
