import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class TrueDynamics(nn.Module):

    def __init__(self, env, hidden_size=200, drop_prob=0.0):
        super().__init__()
        self.env = env
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        self.mask1 = None

    def forward(self, x):
        th = x[:, 0]
        thdot = x[:, 1]
        u = torch.clamp(x[:, 2], -3, 3)
        g = 9.82
        m = 1.0
        l = 1.0
        dt = 0.08
        newthdot = thdot + (-3 * g / (2 * l) * torch.sin(th + np.pi) + 3.0 /
            (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = torch.clamp(newthdot, -8, 8)
        return torch.stack([newth, newthdot], 1)

    def set_sampling(self, sampling=None, batch_size=None):
        if sampling is None:
            raise ValueError('Sampling cannot be None.')
        self.sampling = sampling
        if self.sampling:
            self.mask1 = Variable(torch.bernoulli(torch.zeros(batch_size,
                self.hidden_size).fill_(1 - self.drop_prob)))
            self.mask2 = Variable(torch.bernoulli(torch.zeros(batch_size,
                self.hidden_size).fill_(1 - self.drop_prob)))
            self.mask1 /= 1 - self.drop_prob
            self.mask2 /= 1 - self.drop_prob


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'env': 4}]
