import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardActorNN(nn.Module):

    def __init__(self, in_dim, out_dim, is_discrete):
        super(FeedForwardActorNN, self).__init__()
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)
        self.is_discrete = is_discrete

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        if self.is_discrete:
            output = torch.softmax(self.layer3(activation2), dim=0)
        else:
            output = self.layer3(activation2)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4, 'is_discrete': 4}]
