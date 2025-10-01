import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def fanin_init(size, fanin=None):
    """
    Initilise network weights
    """
    fanin = fanin or size[0]
    v = 1.0 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class DDPGActor(nn.Module):
    """
    Pytorch neural network for Actor model
    """

    def __init__(self, state_dim, action_dim, action_bound, hidden_size,
        init_w=0.003):
        super(DDPGActor, self).__init__()
        self.action_bound = action_bound
        self.l1 = nn.Linear(state_dim, hidden_size)
        self.bn1 = nn.LayerNorm(hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.LayerNorm(hidden_size)
        self.l3 = nn.Linear(hidden_size, action_dim)
        self.init_weights(init_w)
        self.device = torch.device('cuda' if torch.cuda.is_available() else
            'cpu')
        self

    def init_weights(self, init_w):
        self.l1.weight.data = fanin_init(self.l1.weight.data.size())
        self.l2.weight.data = fanin_init(self.l2.weight.data.size())
        self.l3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.bn1(self.l1(x)))
        x = F.relu(self.bn2(self.l2(x)))
        x = torch.tanh(self.l3(x))
        x = x * self.action_bound
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4, 'action_bound': 4,
        'hidden_size': 4}]
