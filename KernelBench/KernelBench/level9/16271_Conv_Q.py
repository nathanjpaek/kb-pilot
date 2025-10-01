import torch
import torch.nn.functional as F
from torch.functional import F
from torch import nn
from typing import *
from torch.nn import functional as F


class Conv_Q(nn.Module):

    def __init__(self, frames, num_actions):
        super(Conv_Q, self).__init__()
        self.c1 = nn.Conv2d(frames, 32, kernel_size=8, stride=4)
        self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.q1 = nn.Linear(3136, 512)
        self.q2 = nn.Linear(512, 16)
        self.q3 = nn.Linear(16, num_actions)
        self.i1 = nn.Linear(3136, 512)
        self.i2 = nn.Linear(512, 16)
        self.i3 = nn.Linear(16, num_actions)

    def forward(self, state):
        c = F.relu(self.c1(state))
        c = F.relu(self.c2(c))
        c = F.relu(self.c3(c))
        q = F.relu(self.q1(c.reshape(-1, 3136)))
        q = F.relu(self.q2(q))
        q = self.q3(q)
        i = F.relu(self.i1(c.reshape(-1, 3136)))
        i = F.relu(self.i2(i))
        i = self.i3(i)
        return q, F.log_softmax(i, dim=1), i

    def encode(self, state):
        with torch.no_grad():
            c = F.relu(self.c1(state))
            c = F.relu(self.c2(c))
            c = F.relu(self.c3(c))
            q = F.relu(self.q1(c.reshape(-1, 3136)))
            q = F.relu(self.q2(q))
            i = F.relu(self.i1(c.reshape(-1, 3136)))
            i = F.relu(self.i2(i))
            return i


def get_inputs():
    return [torch.rand([4, 4, 144, 144])]


def get_init_inputs():
    return [[], {'frames': 4, 'num_actions': 4}]
