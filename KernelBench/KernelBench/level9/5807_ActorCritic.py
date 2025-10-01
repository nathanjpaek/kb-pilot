import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def log_normal_density(x, mean, log_std, std):
    """returns guassian density given x on log scale"""
    variance = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * variance) - 0.5 * np.log(2 * np.pi
        ) - log_std
    log_density = log_density.sum(dim=1, keepdim=True)
    return log_density


class ActorCritic(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(ActorCritic, self).__init__()
        self.logstd = nn.Parameter(torch.zeros(num_actions))
        self.a_fc1 = nn.Linear(num_inputs, hidden_dim)
        self.a_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.a_fc3 = nn.Linear(hidden_dim, num_actions)
        self.a_fc3.weight.data.mul_(0.1)
        self.a_fc3.bias.data.mul_(0.0)
        self.c_fc1 = nn.Linear(num_inputs, hidden_dim)
        self.c_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.c_fc3 = nn.Linear(hidden_dim, 1)
        self.c_fc3.weight.data.mul_(0.1)
        self.c_fc3.bias.data.mul_(0.0)

    def forward(self, x):
        a = F.tanh(self.a_fc1(x))
        a = F.tanh(self.a_fc2(a))
        mean = self.a_fc3(a)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)
        v = F.tanh(self.c_fc1(x))
        v = F.tanh(self.c_fc2(v))
        v = self.c_fc3(v)
        return v, action, mean

    def evaluate(self, x, action):
        v, _, mean = self.forward(x)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        logprob = log_normal_density(action, mean, logstd, std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_inputs': 4, 'num_actions': 4, 'hidden_dim': 4}]
