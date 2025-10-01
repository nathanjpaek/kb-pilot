import torch
import torch.nn as nn


class policy1(nn.Module):

    def __init__(self):
        super(policy1, self).__init__()
        self.sm = nn.Softmax(dim=-1)
        self.actor = nn.Parameter(torch.FloatTensor([-0.35, 0.4, 1]))

    def forward(self):
        mu = self.sm(self.actor)
        return mu


def get_inputs():
    return []


def get_init_inputs():
    return [[], {}]
